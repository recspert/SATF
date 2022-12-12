import argparse

from models.mp import MostPopular
from datareader import entity_names, display_stats, sequential_training_data


def args_parser(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--skip_params', default=True, action="store_false")
    return parser

def create_model(data, verbose=False):
    model = MostPopular(data)
    model.verbose = verbose
    return model

def test_model_factory(dataset, config, args):
    display_stats(dataset)
    train_data = sequential_training_data(dataset[0], *entity_names(args.dataset)[:2], args.maxlen)
    model = create_model(train_data, args.verbose)
    model.build()
    return model