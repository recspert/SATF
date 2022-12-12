import argparse
from collections import defaultdict
import numpy as np
import torch

from polara.tools.random import random_seeds
from polara.evaluation.pipelines import params_to_dict

from models.sasrec import SASRec
from datareader import display_stats, read_dataset
from gridsearch import check_early_stop, try_log_progress, update_best, sweep_grid
from evaluation import evaluate
from utils import WarpSampler, get_wandb, join_str


def args_parser(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    if not test:
        parser.add_argument('--num_epochs', default=200, type=int)
        parser.add_argument('--es_tol', default=0.001, type=float)
        parser.add_argument('--es_max_steps', default=2, type=int)
    return parser 

def create_model(user_train, usernum, itemnum, config, maxlen, device, seed=None):
    model = SASRec(usernum, itemnum, config, maxlen, device).to(device) # no ReLU activation in original SASRec implementation?
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_uniform_(param.data)
        except:
            pass # just ignore those failed init layers
    
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=config['batch_size'], maxlen=maxlen, n_workers=3, seed=seed)
    criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98))
    return model, sampler, criterion, optimizer

def grid_step_skip(param_names):
    def wrapped(params):
        return False
    return wrapped

def run_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion):
    device = model.dev
    for _ in range(num_batch): # tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
        u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
        pos_logits, neg_logits = model(u, seq, pos, neg)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=device), torch.zeros(neg_logits.shape, device=device)
        optimizer.zero_grad()
        indices = np.where(pos != 0)
        loss = criterion(pos_logits[indices], pos_labels[indices])
        loss += criterion(neg_logits[indices], neg_labels[indices])
        if l2_emb != 0:
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.norm(param)
        loss.backward()
        optimizer.step()
    return loss


def train_validate(args, param_grid, param_names, ts=None):    
    dataset, _ = read_dataset(args.dataset, args.time_offsets, stepwise_eval=False, part='validation')
    display_stats(dataset)
    user_train, _, usernum, itemnum = dataset
    assert isinstance(user_train, dict)

    label = join_str(args.model, ts)
    maxlen = args.maxlen
    num_epochs = args.num_epochs
    
    grid_iter = iter(param_grid)
    seed_sequence = iter(random_seeds(len(param_grid), entropy=args.seed))
    target_metric = f'{args.target_metric}@{args.topn}'
    global_best_results = defaultdict(lambda: defaultdict(lambda: -np.inf)) # {metric: {'score': _, 'error': _}}
    
    def evaluate_epoch(epoch, loss, run, model, current_best_results, config):
        # validate model
        model.eval()
        epoch_results = evaluate(model, dataset, args.topn)
        model.train()
        # track progress
        previous_best_score = current_best_results[target_metric]['score'] # fix for early stopping
        if update_best(current_best_results, epoch_results, target_metric):
            config['epoch'] = epoch
            try_log_progress(current_best_results, global_best_results, args, label, config)
        scores = {metric: res['score'] for metric, res in epoch_results.items()}
        run.log({'loss': loss, 'epoch': epoch, **scores})
        # early stopping
        check_early_stop(scores[target_metric], previous_best_score, margin=args.es_tol, max_attempts=args.es_max_steps)
    
    def wrapped():
        params = next(grid_iter)
        if params in sweep_grid(args.sweep, param_names):
            return
        param_config = params_to_dict(param_names, params)
        model, sampler, criterion, optimizer = create_model(user_train, usernum, itemnum, param_config, maxlen, args.device, seed=next(seed_sequence))
        num_batch = len(user_train) // param_config['batch_size'] # tail? + ((len(user_train) % args.batch_size) != 0)
        l2_emb = param_config['l2_emb']
        model.train() # enable model training

        model_best_results = defaultdict(lambda: defaultdict(lambda: -np.inf)) # {metric: {'score': _, 'error': _}}
        epoch_start_idx = 1
        with get_wandb(args.bypass_wandb).init(config=param_config) as run:
            for epoch in range(epoch_start_idx, num_epochs + 1):
                loss = run_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion)
                if epoch % 20 == 0:
                    try:
                        evaluate_epoch(epoch, loss, run, model, model_best_results, param_config)
                    except StopIteration: # early stopping condition met
                        break
            score_best = {metric: res['score'] for metric, res in model_best_results.items()}
            epoch_best = param_config['epoch']
            run.summary.update({'epoch': epoch_best, **score_best}) # display only the best run results once finished
        sampler.close()
    check_early_stop.fail_count = 0 # initialize early stopping
    return wrapped

def summary_params():
    return ['epoch']

def test_model_factory(dataset, config, args):
    display_stats(dataset)
    user_train, _, usernum, itemnum = dataset
    assert isinstance(user_train, dict)
    
    maxlen = args.maxlen
    model, sampler, criterion, optimizer = create_model(user_train, usernum, itemnum, config, maxlen, args.device, seed=args.seed)
    num_batch = len(user_train) // config['batch_size'] # tail? + ((len(user_train) % args.batch_size) != 0)
    l2_emb = config['l2_emb']
    model.train() # enable model training

    num_epochs = config['epoch']
    for _ in range(num_epochs):
        run_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion)
    sampler.close()

    model.eval()
    return model