import argparse
from collections import defaultdict
from tabnanny import verbose
from polara.evaluation.pipelines import params_to_dict, set_config

from models.svd import SVD
from datareader import entity_names, display_stats, read_dataset, sequential_training_data
from gridsearch import try_log_progress, sweep_grid
from evaluation import evaluate
from utils import get_wandb, join_str


def args_parser(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--randomized', default=True, action="store_false")
    return parser

def create_model(data, randomized, verbose=False):
    model = SVD(data, randomized=randomized)
    model.verbose = verbose
    return model

def grid_step_skip(param_names):
    def wrapped(params):
        h = param_names.index('rescaled')
        s = param_names.index('scaling')
        if params[h] and (params[s] == 1.0):
            return True
        return False
    return wrapped


def train_validate(args, param_grid, param_names, ts=None):
    dataset, _ = read_dataset(args.dataset, args.time_offsets, stepwise_eval=False, part='validation')
    display_stats(dataset)
    train_data = sequential_training_data(dataset[0], *entity_names(args.dataset)[:2], args.maxlen)
    model = create_model(train_data, args.randomized, verbose=args.verbose)
    
    grid_iter = iter(param_grid)
    best_results = defaultdict(lambda: defaultdict(float)) # {metric: {'score': _, 'error': _}}
    label = join_str(args.model, ts)
    
    def wrapped():
        params = next(grid_iter)
        if params in sweep_grid(args.sweep, param_names):
            return        
        param_config = params_to_dict(param_names, params)
        set_config(model, param_config)
        with get_wandb(args.bypass_wandb).init(config=param_config) as run:
            model.build()
            results = evaluate(model, dataset, args.topn)
            try_log_progress(results, best_results, args, label, param_config)
            scores = {metric: res['score'] for metric, res in results.items()}
            run.log(scores)
    return wrapped


def test_model_factory(dataset, config, args):
    display_stats(dataset)
    train_data = sequential_training_data(dataset[0], *entity_names(args.dataset)[:2], args.maxlen)
    model = create_model(train_data, args.randomized)
    set_config(model, config)
    model.build()
    return model
