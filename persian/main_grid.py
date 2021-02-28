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
        # save and change logdir
        logdir, timestamp = flags['logdir'], round(time.time())
        flags['logdir'] = logdir.parent / f"{logdir.name}_progress_{timestamp}"

        model = DynSchema(flags)
        hstr = model.as_hstr()
        if len(list(logdir.glob(f'{hstr}_*'))) == 0:
            saved_torch_training(model)

        # move content of flags[logdir] into logdir
        logdir.mkdir(exist_ok=True)
        for p in flags['logdir'].glob('*'):
            p.rename(logdir / p.name)
        flags['logdir'].rmdir()
