import argparse

def dynamic_import(module, cls):
    imp = __import__(module, fromlist=[cls])
    return getattr(imp, cls)

def experiment_builder(parser_experiments=None, verbose=True):
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument(
        '--schema',
        type=str,
        required=True,
        help='for example `torch_dataset`')
    args_main, args_unknown = parser_main.parse_known_args()

    parts = args_main.schema.split('_')[::-1]
    schema_cls = "".join([x[:1].upper() + x[1:] for x in parts]) + 'Schema'
    DynSchema = dynamic_import(f'persian.schemas.{args_main.schema}', schema_cls)

    print(args_main.schema, schema_cls)

    schema_parser = DynSchema.build_parser()
    if parser_experiments is None:
        print('XXX', args_unknown)
        args = schema_parser.parse_args(args=args_unknown)
        args_snd = None
    else:
        args, args_unknown_snd = schema_parser.parse_known_args(args=args_unknown)
        args_snd = parser_experiments.parser_args(args=args_unknown_snd)

    model = DynSchema.from_args(args)
    if verbose:
        print('Running', model.as_hstr())
    return args, args_snd