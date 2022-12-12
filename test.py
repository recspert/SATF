from gridsearch import get_test_config, run_test
from defaults import configure_environment; configure_environment()
from datetime import datetime
from utils import experiment_factory, parse_args


if __name__ == "__main__":
    main_args, specs = parse_args(test=True)
    spec_parser, model_factory, summary_params = experiment_factory(
        main_args.model, ['args_parser', 'test_model_factory', 'summary_params']
    )
    args = spec_parser().parse_args(specs, namespace=main_args)

    test_config = get_test_config(args, summary_params=summary_params)
    ts = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    
    print(f'\nStarted test run at {ts}.\n')
    run_test(model_factory, ts, args, dest_sweep=args.dest_sweep, config=test_config)
