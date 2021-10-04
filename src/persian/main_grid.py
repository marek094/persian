import argparse
import time
from persian.trainer import saved_torch_training


def dynamic_import(module, cls):
    imp = __import__(module, fromlist=[cls])
    return getattr(imp, cls)


def partitioned(gen, nth, ofn):
    r = nth - 1
    return (val for ix, val in enumerate(gen) if (ix % ofn) == r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema',
                        type=str,
                        default='schema_mnist_cnn.CnnMnistSchema')
    parser.add_argument('--nth', type=int, default=1)
    parser.add_argument('--ofn', type=int, default=1)
    args = parser.parse_args()

    *modules, cls = args.schema.split('.')
    DynSchema = dynamic_import(".".join(['schemas'] + modules), cls)

    hgg = DynSchema.hgrid_gen()
    for flags in partitioned(hgg, args.nth, args.ofn):
        logdir_final = flags['logdir'].parent / f"{flags['logdir'].name}_final"
        logdir_final.mkdir(exist_ok=True)

        model = DynSchema(flags)
        hstr = model.as_hstr()
        if len(list(logdir_final.glob(f'{hstr}_*'))) == 0:
            saved_torch_training(model)

        # move relevant content of flags[logdir] into logdir_final
        for p in flags['logdir'].glob(f'{hstr}_*'):
            p.rename(logdir_final / p.name)
