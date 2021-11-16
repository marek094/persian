from persian.schemas.torch_dataset_cnn import CnnDatasetTorchSchema
from persian.errors import *
import persian.models.model_resnet18k as resnet18

import torch

import numpy as np
from collections import defaultdict


class BoundaryCnnSchema(CnnDatasetTorchSchema):
    @staticmethod
    def list_hparams():
        return CnnDatasetTorchSchema.list_hparams() + [
            dict(name='model', type=str, default='resnet18'),
            dict(name='width', type=int, default=64),
            dict(name='save_mode', type=str, default='g127')
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags=flags)

        if self.flags['model'] not in ['resnet18']:
            raise UnknownFlagValueError('Model name is not known')

        if self.flags['h0_decay'] != 0.0:
            assert self.flags['sub_batches'] is None
            raise IncompatibleFlagsError('H0 decay is not allowed')

        self.epoch_num = 0
        self.saving_sched = self._get_saving_schedule()

    def _get_saving_schedule(self):
        mode = self.flags['save_mode'][:1]
        num = int(self.flags['save_mode'][1:])

        if mode not in ['g']:
            raise UnknownFlagValueError('This save_mode value is not known')

        saving_sched = [0]
        if mode == 'g':
            space = np.geomspace(1, self.flags['epochs'], num, endpoint=True)
            saving_sched += np.unique(np.round(space).astype(int)).tolist()
        return saving_sched

    def make_cnn(self):
        return resnet18.make_resnet18k(
            width=self.flags['width'],
            num_classes=10,
        )

    def epoch_range(self):
        for i in super().epoch_range():
            self.epoch_num += 1
            yield i

    def is_to_be_saved(self):
        return self.epoch_num in self.saving_sched

    def _save_featspace(self, featspace):
        path = self.logs_path / f'featspace_{self.epoch_num:04}.npz'
        for k, v in featspace.items():
            featspace[k] = np.asarray(v)
        np.savez_compressed(path, **featspace)

    def _run_batches_train(self, set_name):
        self._run_batches_valid(set_name)
        return super()._run_batches_train(set_name)

    def _run_batches_valid(self, set_name):
        self.model.eval()
        correct, total = 0, 0
        losses_means, weight = defaultdict(float), 0

        featspace = dict(feats=[], labels=[])
        with torch.no_grad():
            for inputs, targets in self.loaders[set_name]:
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                logits, feats = self.model(inputs)
                losses = {
                    'loss/std': self.crit(logits, targets),
                    'loss/top': self._topological_crit(feats),
                }

                if self.is_to_be_saved():
                    featspace['feats'] += feats.cpu().detach().numpy().tolist()
                    featspace['labels'] += targets.cpu().numpy().tolist()

                # loss
                for k, l in losses.items():
                    losses_means[k] += l.item()
                weight += targets.size(0) / self.flags['batch_size']

                # accuracy
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

        for k in losses_means:
            losses_means[k] /= weight

        if self.is_to_be_saved():
            self._save_featspace(featspace)

        return {
            'acc': 100.0 * correct / total,
            'loss_': sum(losses_means.values()),
            **losses_means
        }
