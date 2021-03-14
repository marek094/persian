import argparse

from persian.trainer import saved_torch_training
from persian.main_grid import dynamic_import

if __name__ == "__main__":
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument(
        '--schema',
        type=str,
        default=None,
        help='for example `schema_mnist_cnn.CnnMnistSchema`')
    parser_main.add_argument('--persian', type=str, default=None)
    parser_main.add_argument('--experiments', type=str, default=None)
    args_main, unknown = parser_main.parse_known_args()

    if args_main.schema is not None:
        lib, cls_str = 'schemas', args_main.schema
    elif args_main.persian is not None:
        lib, cls_str = 'persian', args_main.persian
    elif args_main.experiments is not None:
        lib, cls_str = 'experiments', args_main.experiments

    *mds, cls = cls_str.split('.')
    DynSchema = dynamic_import('.'.join([lib] + mds), cls)

    parser = DynSchema.build_parser()
    args = parser.parse_args(args=unknown)

    model = DynSchema.from_args(args)
    print('Running', model.as_hstr())
    saved_torch_training(model, verbose=True)
