from utils import execute_function, get_args, prepare_runtime_args

if __name__ == "__main__":
    args = prepare_runtime_args(get_args())
    main_fn = execute_function(args.method, args.mode)
    main_fn(args)
