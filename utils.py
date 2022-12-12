import os
import sys
import argparse
import importlib
import shutil
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
from multiprocessing import Process, Queue
try:
    import wandb
except ImportError:
    wandb = None

# sampler for batch generation
def random_neq(l, r, s, random_state):
    t = random_state.randint(l, r)
    while t in s:
        t = random_state.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(random_state):
        user = random_state.randint(1, usernum + 1)
        while len(user_train.get(user, [])) <= 1:
            user = random_state.randint(1, usernum + 1)

        user_items = user_train[user]
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_items[-1]
        idx = maxlen - 1

        ts = set(user_items)
        for i in reversed(user_items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts, random_state)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        return (user, seq, pos, neg)

    random_state = np.random.RandomState(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(random_state))

        result_queue.put(zip(*one_batch))


class WarpSampler:
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1, seed=None):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(
                    target=sample_function, 
                    args=(User, usernum, itemnum, batch_size, maxlen, self.result_queue, seed)
                )
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open('data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        user_items = User[user]
        nfeedback = len(user_items)
        if nfeedback < 3:
            user_train[user] = user_items
        else:
            user_train[user] = user_items[:-2]
            user_valid[user] = user_items[-2]
            user_test[user] = user_items[-1]
    return [user_train, user_valid, user_test, usernum, itemnum]

def prepare_sweep(name, sweep=None, project=None, entity=None, **kwargs):
    sweep_config = {
        'name': name,
        'method': 'random', # enables unlimited loop
        'parameters': {} # will be set at runtime
    }
    sweep_id = sweep or wandb.sweep(sweep_config, project=project, entity=entity, **kwargs)
    return sweep_id

def wandb_save_results(result_filepath, res_suffix='result', cfg_suffix='config', grid_suffix='grid', dummy_wandb=False):
    config_filepath = result_filepath.replace(f'{res_suffix}.json', f'{cfg_suffix}.json')
    grid_filepath = result_filepath.replace(f'{res_suffix}.json', f'{grid_suffix}.json')
    if dummy_wandb:
        def check_path(path):
            return "OK" if os.path.exists(path) else "-" 
        print(f'{result_filepath}: {check_path(result_filepath)}')
        print(f'{config_filepath}: {check_path(config_filepath)}')
        print(f'{grid_filepath}: {check_path(grid_filepath)}')
        return
    
    def try_copy(path):
        if os.path.exists(path):
            shutil.copy(path, wandb.run.dir)
    with get_wandb().init():
        _, res_file = os.path.split(os.path.splitext(result_filepath)[0])
        wandb.run.name = res_file
        try_copy(result_filepath)
        try_copy(config_filepath)
        try_copy(grid_filepath)

def save_results(sweep_id, res_file, grid_file=None, bypass_wandb=False):
    get_wandb(sweep_id is None).agent(
        sweep_id,
        function=lambda: wandb_save_results(
            res_file, dummy_wandb=bypass_wandb
        ),
        count=1
    )


def get_wandb(dummy=False):
    if dummy or (wandb is None):
        return DummyWandb()
    return wandb

class DummyWandbRun:
    name = None
    dir = None

class DummyWandbApi:
    runs = []

    @property
    def sweep(self, *args, **kwargs):
        return self

class DummyWandb:
    summary = {}

    def __init__(self):
        self.run = DummyWandbRun()
    
    def sweep(self, *args, **kwargs):
        pass
    
    def Api(self):
        return DummyWandbApi()

    @contextmanager
    def init(self, config=None):
        try:
            yield self
        finally:
            pass

    def log(self, *args, **kwargs):
        pass

    def agent(self, sid, function, count=None, **kwargs):
        if count is None:
            count = 1
        for i in range(count):
            print(f'\n=== Experiment run: {i+1} ===\n')
            function()

def import_source_as_module(source_path):
    'Importing module from a specified path.'
    'See https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly'
    _, file_name = os.path.split(source_path)
    module_name = os.path.splitext(file_name)[0]
    module_spec = importlib.util.spec_from_file_location(module_name, source_path)
    if module_name in sys.modules:
        print(f'Module {module_name} is already imported!')
        module = sys.modules[module_name]
    else:
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    return module

def parse_args(test=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str)
    parser.add_argument('--dataset', required=True, type=str)
    parser.add_argument('--time_offsets', required=True, type=str)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--target_metric', default='NDCG', type=str)
    parser.add_argument('--topn', default=10, type=int)
    parser.add_argument('--res_dir', default=None, type=str)
    parser.add_argument('--grid_steps', default=0, type=int) # 0 means run all grid points
    parser.add_argument('--sweep', default=None, type=str)
    parser.add_argument('--name', default=None, type=str)
    parser.add_argument('--bypass_wandb', default=False, action="store_true")
    if not test:
        parser.add_argument('--grid_config', required=True, type=str)
    else:
        parser.add_argument('--dest_sweep', default=None, type=str)
        parser.add_argument('--run_id', default=None, type=str)
        parser.add_argument('--test_config', default=None, type=str)
    [args, extras] = parser.parse_known_args()
    check_args(args)
    return args, extras

def check_args(args):
    if not args.res_dir:
        args.res_dir = args.model
    if not args.sweep:
        args.bypass_wandb = True
    assert args.grid_steps >= 0


def join_str(*args):
    return '_'.join(filter(None, args))


def experiment_factory(model_name, routines):
    experiments = {
        'la-satf': 'lsatf',
        'ga-satf': 'gsatf',
        'sasrec': 'sasrec',
        'svd': 'svd',
        'puresvd': 'svd',
        'mp': 'mp',
        'mostpopular': 'mp',
        'rnd': 'rnd',
        'random': 'rnd',
    }
    try:
        module_name = f'experiment.{experiments[model_name.lower()]}'
    except KeyError:    
        raise ValueError(f'Unrecognized model name {model_name}.')
    module = importlib.import_module(module_name)
    return [getattr(module, routine, None) for routine in routines]