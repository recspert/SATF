from collections import defaultdict
from math import sqrt
import numpy as np
import pandas as pd


def sample_ci(scores, coef=2.776):
    n = len(scores)
    if n < 2: # unable to estimate ci
        return np.nan
    return coef * np.std(scores, ddof=1) / sqrt(n)

METRICS = ['NDCG', 'MRR', 'HR', 'COV']

def hr_score(hit_idx):
    return 1.0

def mrr_score(hit_idx):
    return 1.0 / (hit_idx + 1)

def ndcg_score(hit_idx):
    # ideal DCG is 1/log2(2) = 1
    return 1.0 / np.log2(hit_idx + 2)

def get_metric_func(metric_name):
    metric = metric_name.lower()
    if metric == 'hr':
        return hr_score,
    if metric == 'mrr':
        return mrr_score
    if metric == 'ndcg':
        return ndcg_score
    raise ValueError('Unrecognized metric')


def evaluate_step(model, train, test_seq, topn):
    hr = []
    mrr = []
    ndcg = []
    unique_items = set()
    seen_test = defaultdict(list)
    for user, test_item in test_seq:
        if (user not in train) and (user not in seen_test):
            seen_test[user] = [test_item]
            continue

        seq = train.get(user, []) + seen_test.get(user, [])
        hit_index, predicted_items = model.check_hit(test_item, seq, topn)
    
        try:
            hit_index = hit_index.item() # we expect only 1 item here
        except ValueError: # empty index or more then 1 item (which is incorrect)
            hr_inc = mrr_inc = ndcg_inc = 0
        else:
            hr_inc = hr_score(hit_index)
            mrr_inc = mrr_score(hit_index)
            ndcg_inc = ndcg_score(hit_index)
        hr.append(hr_inc)
        mrr.append(mrr_inc)
        ndcg.append(ndcg_inc)

        seen_test[user].append(test_item) # extend seen items for next step prediction
        unique_items = unique_items.union(predicted_items)
    
    scores = {
        f'NDCG@{topn}': np.mean(ndcg),
        f'MRR@{topn}': np.mean(mrr),
        f'HR@{topn}': np.mean(hr),
        f'COV@{topn}': len(unique_items),
    }
    sqerrors = {
        f'NDCG@{topn}': np.mean((ndcg - scores[f'NDCG@{topn}'])**2) / (len(ndcg) - 1),
        f'MRR@{topn}': np.mean((mrr - scores[f'MRR@{topn}'])**2) / (len(mrr) - 1),
        f'HR@{topn}': np.mean((hr - scores[f'HR@{topn}'])**2) / (len(hr) - 1),
    }
    return scores, sqerrors

def evaluate(model, dataset, topn):
    train, test_data, _, itemnum = dataset
    results = defaultdict(lambda: defaultdict(list)) # {metric: {'scores': [], 'squared_errors': []}}
    if isinstance(test_data, (list, tuple)):
        test_data = pd.Series({0: test_data})
    
    for step, test_seq in test_data.iteritems():
        step_scores, step_sqerr = evaluate_step(model, train, test_seq, topn)
        for metric, score in step_scores.items():
            if metric.startswith('COV'):
                score /= itemnum
                error = None # COV error is computed across steps, not per step
            else:
                error = step_sqerr[metric]
            results[metric]['scores'].append(score)
            results[metric]['squared_errors'].append(error)
    
    averaged_results = defaultdict(dict) # {metric: {'score': _, 'error': _}}
    for metric, res in results.items():
        averaged_results[metric]['score'] = np.mean(res['scores'])
        if metric.startswith('COV'):
            averaged_results[metric]['error'] = sample_ci(res['score'])
        else:
            averaged_results[metric]['error'] = sqrt(sum(res['squared_errors'])) / len(res['squared_errors'])
    return averaged_results