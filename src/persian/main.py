import argparse

def dynamic_import(module, cls):
    imp = __import__(module, fromlist=[cls])
    return getattr(imp, cls)

def experiment_builder(parser_experiments=None, verbose=True, args=None):
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument(
        '--schema',
        type=str,
        required=True,
        help='for example `torch_dataset`')
    parser_main.add_argument('--print_help', action='store_true')
    args_main, args_unknown = parser_main.parse_known_args(args=args)

    parts = args_main.schema.split('_')[::-1]
    schema_cls = "".join([x[:1].upper() + x[1:] for x in parts]) + 'Schema'
    DynSchema = dynamic_import(f'persian.schemas.{args_main.schema}', schema_cls)

    # print('From', args_main.schema, 'importing', schema_cls)

    parser_schema = DynSchema.build_parser()
    if args_main.print_help:
        parser_main.print_help()
        parser_schema.print_help()
        if parser_experiments is not None:
            parser_experiments.print_help()

    if parser_experiments is None:
        args = parser_schema.parse_args(args=args_unknown)
        args_snd = None
    else:
        args, args_unknown_snd = parser_schema.parse_known_args(args=args_unknown)
        args_snd = parser_experiments.parser_args(args=args_unknown_snd)

    model = DynSchema.from_args(args)
    if verbose:
        print('Run:', model.as_hstr())
    return model, args_snd