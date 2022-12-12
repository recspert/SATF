# General instructions
For reproducing our work run the commands described below. If you just want to see final optimal configurations and the scores (reported in the paper), please scroll down to the [this section](#the-reported-results-and-corresponding-configurations). For the software requirements and the list of python packages see [Requirements](#requirements) section (please follow the described steps for setting an environment before proceeding).

## Downloading and preparing data
You need to run this command once in order to download and preprocess datasets the same way we did in our paper:
```
python data/prepare.py
```
It automatically selects the necessary columns, convert time info into the unified format across all datasets, performs p-core filtering (for Steam) and saves data into compact gz files in the `data` folder. Note, **it may take a long time to process all 4 datasets**, approximately 40+ minutes.

## Hyper-parameter tuning
To launch a grid-search experiments run:
```
python tune.py --model=<model name> --dataset=<dataset_name> --time_offset=<valid-test-time> --maxlen=<max-sequence-length> --grid_config=<name-of-config-file> --bypass_wandb
```

### Possible values of arguments (case-insensitive)
- `<model name>`
  - MP or MostPopular
  - SVD or PureSVD
  - GA-SATF
  - LA-SATF
  - SASRec
- `<dataset_name>` / `<valid-test-time>` / `<max-sequence-length>`
  - ML-1M / "4months-4months" / 200
  - AMZ-B / "3weeks-3weeks" / 50
  - AMZ-G / "6weeks-6weeks" / 50
  - Steam / "1days-2days" / 50
- `<name-of-config-file>` (files must be stored in the `grids` folder)
  - gsatf_grid
  - lsatf_grid or lsatf_grid_ml1m
  - sasrec_grid
  - svd_grid
  - svdN_grid

**Important note**: as the multilinear rank values and the attention window size for the LA-SATF model depend on the dataset, you'll need to adjust the grid configuration file at settings accordingly. By default, the grid configuration provided in `lsatf_grid` is suitable for all datasets except `ML-1M`. For the ML-1M dataset, you have to provide the `lsatf_grid_ml1m` option to the `--grid_config=` argument.

You can also specify your onw grid-search config files and place them into the `grids` folder. Or, alternatively, modify the default ones listed above.

If you have `wanbd` login and want to store results in the cloud, then remove `--bypass_wandb` switch. You may need to properly initialize your wandb project/entity settings or set `sweep_args` dictionary in the `tune.py` module. You can also explicitly specify an existing sweep to report to by passing `--sweep=<sweep-id>`.

You can additionally pass `--grid_steps=n` to run over only `n` random grid points. If this parameter is not set, the program will run until parameter grid is exhausted or execution is interrupted by user.

Once the grid is exhausetd (or user interrupted execution by pressing Ctrl+C), the program will attempt to run final test based on the best found config and report the results. You can also run test separately using any valid configuration as described below.

## Manually running tests
If you want to see scores corresponding to a specific configuration, you can run:

```
python test.py --model=<model name> --dataset=<dataset_name> --time_offset=<valid-test-time> --maxlen=<max-sequence-length> --test_config=<path-to-config-file> --bypass_wandb
```

Most of the aruments here are the same and should correspond to experiments from the tuning phase. If you have a local config file, then it must be specified via `<path-to-config-file>`. You can also specify `--sweep` (don't forget to remove `--bypass_wandb`), in which case the program will analyze all runs in the specified sweep and take configuration corresponding the highest target metric score (`NDCG@10` by default). You can also specify an exact run from the sweep by passing `--run_id=<sweeep-run-id>`.

## Examples
- Running grid search (locally) for `LA-SATF` model on `Amazon Beauty` dataset:  
  `python tune.py --model=LA-SATF --dataset=amz-b --time_offset=3weeks-3weeks --maxlen=50 --grid_config="lsatf_grid" --bypass_wandb`
  - if you let the program to exhaust entire grid or if you interrupt its execution at som moment (by hitting Ctrl+C once), it will automatically perform testing on the best found hyper-parameters configuration and report the results.
- Running test (locally) with simple popularity-based model, which does not require any confguration (hence, `--test_config` argument is omitted):  
  `python test.py --model=MP --dataset=amz-b --time_offset=3weeks-3weeks --maxlen=50  --bypass_wandb`

# The reported results and corresponding configurations
## Hyper-parameters
|           |                  |     AMZ-B |     AMZ-G |     ML-1M |    Steam |
|:----------|:-----------------|----------:|----------:|----------:|---------:|
| GA-SATF   | scaling          |    0.2    |    0.4    |    0      |    0     |
|           | pos_rank         |    2      |   20      |   12      |   10     |
|           | item_rank        |  600      |  800      |  100      |  400     |
|           | user_rank        |  800      |  900      |  300      |  700     |
|           | rescaled         |  False    |  False    |  True     |  False   |
|           | attention_decay  |    1      |    0      |    1.2    |    0     |
|           | num_iters        |    2      |    2      |    6      |    1     |
| LA-SATF   | scaling          |    0.2    |    0.2    |    0.2    |    0     |
|           | item_rank        |  600      |  600      |  200      |  900     |
|           | user_rank        |  700      |  500      |  600      |  500     |
|           | rescaled         |  False    |  False    |  True     |  False   |
|           | attention_decay  |    0      |    0      |    1      |    1     |
|           | num_iters        |    3      |    3      |    4      |    2     |
|           | attention_rank   |    1      |    1      |   20      |    1     |
|           | sequences_rank   |    2      |    2      |   20      |    2     |
|           | attention_window |    2      |    2      |   40      |    2     |
| PureSVD   | scaling          |    1      |    1      |    1      |    1     |
|           | rank             |  800      | 1500      |  100      |  100     |
| PureSVD-N | scaling          |    0.6    |    0.2    |    0.0    |    0.0   |
|           | rank             | 2000      | 1000      |  800      | 1500     |
|           | rescaled         | False     | False     | False     | False    |
| SASRec    | lr               |    0.0001 |    0.0001 |    0.0001 |    1e-05 |
|           | l2_emb           |    0      |    0      |    0      |    0     |
|           | num_heads        |    1      |    1      |    1      |    1     |
|           | batch_size       |  128      |  512      |  128      |  256     |
|           | num_blocks       |    1      |    2      |    2      |    3     |
|           | dropout_rate     |    0.2    |    0.2    |    0.4    |    0.2   |
|           | hidden_units     |  256      |  768      |  256      |  512     |
|           | epoch            |  100      |  100      |  120      |   80     |

## Final scores
|       | metric@10   | GA-SATF      | LA-SATF      | MP           | PureSVD      | PureSVD-N    | RND          | SASRec       |
|:------|:------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|:-------------|
| amz-b | COV         | 0.182        | 0.608        | 0.007        | 0.251        | 0.615        | 0.985        | 0.611        |
|       | HR          | 0.079+-0.004 | 0.114+-0.005 | 0.004+-0.001 | 0.082+-0.004 | 0.087+-0.004 | 0.001+-0.000 | 0.100+-0.004 |
|       | NDCG        | 0.043+-0.002 | 0.067+-0.003 | 0.002+-0.000 | 0.046+-0.002 | 0.047+-0.002 | 0.000+-0.000 | 0.055+-0.003 |
| amz-g | COV         | 0.241        | 0.426        | 0.008        | 0.467        | 0.631        | 0.983        | 0.700        |
|       | HR          | 0.074+-0.004 | 0.092+-0.004 | 0.003+-0.001 | 0.070+-0.004 | 0.101+-0.004 | 0.001+-0.000 | 0.094+-0.004 |
|       | NDCG        | 0.046+-0.003 | 0.052+-0.003 | 0.002+-0.000 | 0.042+-0.002 | 0.058+-0.003 | 0.000+-0.000 | 0.055+-0.003 |
| ml-1m | COV         | 0.288        | 0.511        | 0.038        | 0.187        | 0.275        | 1.000        | 0.503        |
|       | HR          | 0.112+-0.004 | 0.132+-0.004 | 0.000+-0.000 | 0.060+-0.003 | 0.061+-0.003 | 0.004+-0.001 | 0.134+-0.004 |
|       | NDCG        | 0.061+-0.002 | 0.072+-0.003 | 0.000+-0.000 | 0.029+-0.002 | 0.030+-0.002 | 0.002+-0.000 | 0.069+-0.002 |
| steam | COV         | 0.047        | 0.368        | 0.018        | 0.070        | 0.438        | 0.997        | 0.080        |
|       | HR          | 0.013+-0.001 | 0.091+-0.003 | 0.000+-0.000 | 0.039+-0.002 | 0.084+-0.003 | 0.001+-0.000 | 0.115+-0.004 |
|       | NDCG        | 0.007+-0.001 | 0.047+-0.002 | 0.000+-0.000 | 0.020+-0.001 | 0.043+-0.002 | 0.001+-0.000 | 0.060+-0.002 |

# Requirements
## mamba
We use `mamba` package manager (based on `conda`) with the default `conda-forge` channel for installing Python packages. The easiest way to get it is to download and install `mamba-forge` distribution from [here](https://github.com/conda-forge/miniforge#mambaforge).

## main python packages
When mamba is installed, you can recreate our einvironment by running:
```
 mamba env create -f environment.yml
```
This will create a new environment named `satf`. The `environment.yml` file with all the needed dependencies is included in this repository.

## extra packages
We use [`Polara` framework]() for orchestrating some parts of the experiments. You can install it into your enviroment by running:
```
conda activate satf
pip install --no-cache-dir --upgrade git+https://github.com/evfro/polara.git@develop#egg=polara
```
