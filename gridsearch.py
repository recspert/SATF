import json
import os
from polara.evaluation.pipelines import random_grid
from utils import get_wandb, prepare_sweep, import_source_as_module, join_str, save_results
from datareader import read_dataset
from evaluation import evaluate

class GridSearchError(Exception): pass

def validate_ranges(config):
    '''handle generators/iterators like range(10)'''
    return {key: list(rng) for key, rng in config.items()}

def save_config(dir, dataset_name, label, params, suffix=None, dest_dir='results'):
    folder_path = os.path.join(dest_dir, dataset_name, dir)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    config_name = join_str(label, dataset_name, suffix)
    config_file = os.path.join(folder_path, f'{config_name}.json')
    with open(config_file, 'w') as file:
        if not isinstance(params, dict):
            config = vars(params)
        else:
            config = params
        json.dump(config, file, indent=4)
    return config_file

def load_config(dir, dataset_name, label, suffix=None, source_dir='results'):
    folder_path = os.path.join(source_dir, dataset_name, dir)
    config_name = join_str(label, dataset_name, suffix)
    config_file = os.path.join(folder_path, f'{config_name}.json')
    with open(config_file, 'r') as file:
        config = json.loads(file.read())
    return config

def sweep_grid(sweep_id, param_names):
    param_grid = set()
    if not sweep_id:
        return param_grid
    sweep = get_wandb().Api().sweep(sweep_id)
    for run in sweep.runs:
        if run.state in ['running', 'finished']:
            config = json.loads(run.json_config, object_hook=lambda x: x.get('value', x))
            try:
                params = tuple(config[name] for name in param_names)
            except KeyError as key: # handle invalid runs, e.g. with no config logged
                print(f'{key} is not found in run {run.name} config of sweep {sweep_id}. Skipping.')
                continue
            param_grid.add(params)
    return param_grid

def run_grid_search(grid_step_call, args, grid_skip=None, sweep_args=None, ts=None):
    grid_config = import_config(args.grid_config, configs_dir='grids')
    save_config(args.res_dir, args.dataset, join_str(args.model, ts), validate_ranges(grid_config), suffix='grid')
    skip_config = None
    if grid_skip:
        skip_config = grid_skip(list(grid_config.keys()))
    param_grid, param_names = random_grid(grid_config, n=0, skip_config=skip_config)
    grid_cache = sweep_grid(args.sweep, param_names)
    if grid_cache:
        starting_size = len(param_grid)
        param_grid.difference_update(grid_cache)
        print(f'{starting_size - len(param_grid)} known grid-search configurations are filtered.\n')
    if not param_grid:
        raise GridSearchError('No grid points left. Terminating')

    sweep_id = None
    if not args.bypass_wandb:
        if sweep_args is None:
            sweep_args = {}
        experiment_name = args.name or sweep_args.get('project', args.model)
        sweep_id = prepare_sweep(join_str(experiment_name, args.dataset, args.maxlen), sweep=args.sweep,  **sweep_args)
    
    max_grid_steps = min(args.grid_steps or len(param_grid), len(param_grid))
    print(f'Total number of configurations to run: {max_grid_steps}')
    try:
        get_wandb(args.bypass_wandb).agent(
            sweep_id,
            function=grid_step_call(args, param_grid, param_names, ts=ts),
            count=max_grid_steps
        )
    except KeyboardInterrupt:
        pass # allow running test if experiment is interrupted by user
    return sweep_id


def check_early_stop(target_score, previous_best, margin=0, max_attempts=1):
    if (margin >= 0) and (target_score < previous_best + margin):
        check_early_stop.fail_count += 1
    else:
        check_early_stop.fail_count = 0
    if check_early_stop.fail_count >= max_attempts:
        print('Interrupted due to early stopping condition.')
        raise StopIteration


def print_scores(scores, errors=None):
    for metric, score in scores.items():
        try:
            err_str = f"{errors[metric]:.4f}"
        except:
            err_str = "nan"
        if isinstance(score, dict):
            err_str = f"{score['error']:.4f}"
            score = score['score']
        score_str = f"{metric}: {score:.4f}"
        print(f"{score_str} +- {err_str}")

def update_best(best_results, results, target_metric):
    updated = results[target_metric]['score'] > best_results[target_metric]['score']
    if updated:
        best_results.update(results)
    return updated

def try_log_progress(results, best_results, args, label, config):
    target_metric = f'{args.target_metric}@{args.topn}'
    if update_best(best_results, results, target_metric):
        save_config(args.res_dir, args.dataset, label, config, suffix='config')
        print('==========================')
        print('Current best:')
        print_scores(best_results)
        print('Achieved at:')
        print(config)
        print('==========================')


def import_config(name, configs_dir='configs'):
    module_path = os.path.join(configs_dir, f'{name}.py')
    module = import_source_as_module(module_path)
    return module.grid_config


def get_test_config(args, summary_params=None):
    config = {}
    if args.test_config:
        with open(args.test_config, 'r') as file:
            config = json.loads(file.read())
    elif args.sweep:
        sweep = get_wandb().Api().sweep(args.sweep)
        if args.run_id:
            for run in sweep.runs:
                if run.id == args.run_id:
                    test_run = run
                    break
        else:
            target_metric = f'{args.target_metric}@{args.topn}'
            best_score = float('-inf')
            for run in sweep.runs:
                run_score = run.summary.get(target_metric, float('-inf'))
                if run_score > best_score:
                    test_run = run
                    best_score = run_score
        
        print(f'Using config from run {test_run.id} ({test_run.name}).')
        config = json.loads(test_run.json_config, object_hook=lambda x: x.get('value', x))
    
        if summary_params:
            for name in summary_params():
                config[name] = test_run.summary[name]
    return config


def run_test(model_factory, ts, args, dest_sweep=None, config=None):
    label = join_str(args.model, ts)
    if not config: # when test is called from tune.py or when model has no params
        try:
            args.skip_params # for parameterless models, e.g. MP
        except AttributeError:
            config = load_config(args.res_dir, args.dataset, label, suffix='config')
    
    _, dataset = read_dataset(args.dataset, args.time_offsets, stepwise_eval=False, part='test')
    model = model_factory(dataset, config, args)
    test_results = evaluate(model, dataset, args.topn)
    
    scores = {metric: res['score'] for metric, res in test_results.items()}
    errors = {metric: res['error'] for metric, res in test_results.items()}
    res_file = save_config(args.res_dir, args.dataset, label, {'scores': scores, 'errors': errors}, suffix='result')
    print_scores(scores, errors)
    # upload results to wandb (or just show paths to files)
    save_results(dest_sweep, res_file, bypass_wandb=args.bypass_wandb)
