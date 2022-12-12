from defaults import configure_environment; configure_environment()
from datetime import datetime
from gridsearch import run_grid_search, run_test
from utils import experiment_factory, parse_args


if __name__ == "__main__":
    main_args, model_specs = parse_args()
    grid_step_call, grid_skip, spec_parser, model_factory = experiment_factory(
        main_args.model, ['train_validate', 'grid_step_skip', 'args_parser', 'test_model_factory']
    )
    args = spec_parser().parse_args(model_specs, namespace=main_args)

    sweep_args = dict(
        project = f"{args.model}-tuning",
        entity = "recsys"
    )

    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print(f'\nExperiment start time: {ts}\n')

    # perform grid search and save results
    print('\nRunning grid search\n')
    sweep_id = run_grid_search(
        grid_step_call,
        args,
        grid_skip=grid_skip,
        sweep_args=sweep_args,
        ts=ts
    )
    # perform test score evaluation
    print('\nRunning final test\n')
    run_test(model_factory, ts, args, dest_sweep=sweep_id)