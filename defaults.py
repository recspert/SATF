import os


num_threads = '' # empty string means use all cores
wandb_silent = 'true'


def configure_environment():
    os.environ['WANDB_SILENT'] = wandb_silent
    if num_threads:
        os.environ['OMP_NUM_THREADS'] = num_threads
        os.environ['MKL_NUM_THREADS'] = num_threads
        os.environ['NUMBA_NUM_THREADS'] = num_threads