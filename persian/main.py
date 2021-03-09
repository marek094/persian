import argparse

from persian.trainer import saved_torch_training
from persian.main_grid import dynamic_import

if __name__ == "__main__":
    parser_main = argparse.ArgumentParser()
    parser_main.add_argument('--schema',
                             type=str,
                             default='schema_mnist_cnn.CnnMnistSchema')
    parser_main.add_argument('--persian', '-p', action='store_true')
    args_main, unknown = parser_main.parse_known_args()

    lib = 'persian' if args_main.persian else 'schemas'

    *mds, cls = args_main.schema.split('.')
    DynSchema = dynamic_import('.'.join[lib] + mds), cls)

    parser = DynSchema.build_parser()
    args = parser.parse_args(args=unknown)

    model = DynSchema.from_args(args)
    print('Running', model.as_hstr())
    saved_torch_training(model, verbose=True)
