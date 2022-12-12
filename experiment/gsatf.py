import argparse
from collections import defaultdict

from polara.evaluation.pipelines import params_to_dict, set_config

from models.gsatf import SequentialTensor, valid_mlrank
from datareader import display_stats, entity_names, read_dataset, sequential_training_data
from gridsearch import check_early_stop, try_log_progress, update_best, sweep_grid
from evaluation import evaluate
from utils import get_wandb, join_str


def args_parser(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--verbose', default=False, action="store_true")
    if not test:
        parser.add_argument('--max_iters', default=20, type=int)
        parser.add_argument('--es_tol', default=0.001, type=float)
        parser.add_argument('--es_max_steps', default=2, type=int)
    return parser

def create_model(data, num_iters, seed=None, verbose=False):
    model = SequentialTensor(data, randomized=True)
    model.num_iters = num_iters
    model.parallel_ttm = [False, False, True]
    model.seed = seed
    model.verbose = verbose
    return model

def grid_step_skip(param_names):
    def wrapped(params):
        urank = param_names.index('user_rank')
        irank = param_names.index('item_rank')
        prank = param_names.index('pos_rank')
        mlrank = (params[urank], params[irank], params[prank])
        scaling = params[param_names.index('scaling')]
        skip_scaling =  params[param_names.index('rescaled')] and (scaling == 1.0)        
        allowed = valid_mlrank(mlrank) and not skip_scaling
        return not allowed
    return wrapped

def train_validate(args, param_grid, param_names, ts=None):
    dataset, _ = read_dataset(args.dataset, args.time_offsets, stepwise_eval=False, part='validation')
    display_stats(dataset)
    train_data = sequential_training_data(dataset[0], *entity_names(args.dataset)[:2], args.maxlen)
    model = create_model(train_data, args.max_iters, seed=args.seed, verbose=args.verbose)
    
    grid_iter = iter(param_grid)
    target_metric = f'{args.target_metric}@{args.topn}'
    global_best_results = defaultdict(lambda: defaultdict(float)) # {metric: {'score': _, 'error': _}}

    def evaluation_callback(run, current_best_results, config):
        '''Implements early stopping for ALS procedure.'''
        label = join_str(args.model, ts)
        def iterative_call(step, core_norm, factors):
            # validate model
            model.store_factors(*factors)
            iter_results = evaluate(model, dataset, args.topn)
            # track progress
            previous_best_score = current_best_results[target_metric]['score'] # fix for early stopping            
            if update_best(current_best_results, iter_results, target_metric):
                config['num_iters'] = step + 1
                try_log_progress(current_best_results, global_best_results, args, label, config)
            scores = {metric: res['score'] for metric, res in iter_results.items()}
            run.log({'num_iters': step + 1, 'core_norm': core_norm, **scores})
            # early stopping condition
            check_early_stop(scores[target_metric], previous_best_score, margin=args.es_tol, max_attempts=args.es_max_steps)
        return iterative_call
    
    def wrapped():
        params = next(grid_iter)
        if params in sweep_grid(args.sweep, param_names):
            return        
        param_config = params_to_dict(param_names, params)
        set_config(model, param_config)
        rank_names = ['user_rank', 'item_rank', 'pos_rank']
        model.mlrank = tuple(param_config[name] for name in rank_names)
        model_best_results = defaultdict(lambda: defaultdict(float)) # {metric: {'score': _, 'error': _}}
        with get_wandb(args.bypass_wandb).init(config=param_config) as run:
            model.build(callback=evaluation_callback(run, model_best_results, param_config))
            score_best = {metric: res['score'] for metric, res in model_best_results.items()}
            run.summary.update({'num_iters': param_config['num_iters'], **score_best}) # display only the best run results once finished
    check_early_stop.fail_count = 0 # initialize for checking early stopping condition
    return wrapped

def summary_params():
    return ['num_iters']

def test_model_factory(dataset, config, args):
    display_stats(dataset)
    train_data = sequential_training_data(dataset[0], *entity_names(args.dataset)[:2], args.maxlen)
    model = create_model(train_data, num_iters=None, seed=args.seed, verbose=args.verbose) # num_iters is set via test config
    set_config(model, config) # also sets the optimal number of ALS iterations
    rank_names = ['user_rank', 'item_rank', 'pos_rank']
    model.mlrank = tuple(config[name] for name in rank_names)
    model.build(callback=lambda step, core_norm, *args, **kwargs: print(f'Step {step+1} core norm: {core_norm}')) # exhaust num_iters iterations
    return model