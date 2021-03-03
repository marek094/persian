from persian.schema_protocol import Schemable

from pathlib import Path


def _vprint(verbose):
    return print if verbose else lambda *a, **b: None


def validated_training(model: Schemable,
                       verbose=True,
                       kTRAIN='TRAIN',
                       kVALID='VALID'):
    vprint = _vprint(verbose)
    model.prepare_dataset(kTRAIN)
    model.prepare_dataset(kVALID)

    model.prepare_model()
    model.prepare_criterium()
    for ep in model.epoch_range():
        vprint(f'Epoch {ep}')
        vprint('-' * 10)

        model.run_batches(kTRAIN)
        vprint('Train\t' + model.metrics_report(kTRAIN))
        model.update_infoboard()

        model.run_batches(kVALID)
        vprint('Valid\t' + model.metrics_report(kVALID))
        model.update_infoboard()

    return model


def saved_torch_training(model: Schemable,
                         verbose=True,
                         kTRAIN='TRAIN',
                         kVALID='VALID'):
    import torch as T
    vprint = _vprint(verbose)

    ks = [kTRAIN, kVALID]
    for k in ks:
        model.prepare_dataset(k)

    model.prepare_model()
    model.prepare_criterium()

    for ep in model.epoch_range():
        vprint(f'Epoch {ep}')

        for k in ks:
            model.run_batches(k)
            model.update_infoboard()
            vprint(k + '\t' + model.metrics_report(k))

        params = model.pack_model_params()

        path = model.flags['logdir'] / f'{model.as_hstr()}_{ep:04}.pt'
        T.save(params, path)

        for k in [kTRAIN, kVALID]:
            path_report = model.flags[
                'logdir'] / f'{model.as_hstr()}_{ep}_{k}.txt'
            with path_report.open('w') as f:
                print(model.metrics_report(k), file=f)
