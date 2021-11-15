from collections import defaultdict
from typing import Iterator

from persian.schemas.torch_dataset import DatasetTorchSchema
from persian.errors.flags_incompatible import IncompatibleFlagsError
from persian.errors.value_flag_unknown import UnknownFlagValueError
from persian.global_h0_regularizer import GlobalH0Regularizer

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import numpy as np


class CnnDatasetTorchSchema(DatasetTorchSchema):
    @staticmethod
    def list_hparams():
        return DatasetTorchSchema.list_hparams() + [
            dict(name='epochs', type=int, default=80),
            dict(name='lr', type=float, default=0.1, range=(0.01, 0.21, 0.01)),
            dict(name='optim', type=str, default='sgd'),
            dict(name='sched', type=str, default=None),
            dict(name='w_decay', type=float, default=5e-4),
            # dict(name='lr_decay', type=int, default=1000),
            # dict(name='width', type=int, default=10, range=(2, 80, 4)),
            dict(name='h0_decay', type=float, default=0.0),
            dict(name='h0_dens', type=float, default=0.7),
            dict(name='h0_global', type=float, default=0.0)
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags)
        if flags['h0_decay'] != 0.0 and flags['sub_batches'] is None:
            raise IncompatibleFlagsError(
                "Non-zero `h0_decay` requires `sub_batches` enabled")

    def make_cnn(self):
        raise NotImplementedError()

    def prepare_model(self):
        self.model = self.make_cnn()
        self.model = self.model.to(self.dev)

    def prepare_criterium(self):
        if self.flags['optim'] == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(),
                                          lr=self.flags['lr'],
                                          weight_decay=self.flags['w_decay'])
        elif self.flags['optim'] == 'sgd':
            self.optim = torch.optim.SGD(
                [
                    dict(
                        params=self.model.feat_ext.parameters(),
                        weight_decay=self.flags['w_decay'],
                    ),
                    dict(
                        params=self.model.cls.parameters(),
                        weight_decay=self.flags['w_decay'],
                    )
                ],
                lr=self.flags['lr'],
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise UnknownFlagValueError(f"Unknown value of `optim`")

        if self.flags['sched'] == 'None':
            self.scheduler = StepLR(self.optim, step_size=1, gamma=1.0)
        elif self.flags['sched'] == 'cos':
            self.scheduler = CosineAnnealingLR(self.optim,
                                               T_max=self.flags['epochs'],
                                               eta_min=0,
                                               last_epoch=-1)
        else:
            raise UnknownFlagValueError("Unknown value of `sched`")

        self.crit = nn.CrossEntropyLoss()

        self.global_h0_regularizer = None
        if self.flags['h0_global'] > 0.0:
            print('COEF.', self.flags['h0_global'])
            self.global_h0_regularizer = GlobalH0Regularizer(
                data_size=self.flags['train_size'],
                feat_dim=128,
                nclasses=10,
                device=self.dev,
                decay=self.flags['h0_global'],
            )

    def epoch_range(self) -> Iterator[int]:
        for i in range(self.flags['epochs']):
            self.epoch_i = i
            yield i

    def run_batches(self, set_name):
        if set_name == 'TRAIN':
            self.metrics[set_name] = self._run_batches_train(set_name)
        else:
            self.metrics[set_name] = self._run_batches_valid(set_name)

    def run_inference(self, input):
        return self.model([input])[0]

    def _run_batches_train(self, set_name):
        self.model.train()
        correct, total = 0, 0
        losses_means, weight = defaultdict(float), 0
        # glob_features = []
        for idcs, inputs, targets in self.loaders[set_name]:
            inputs, targets = inputs.to(self.dev), targets.to(self.dev)
            self.optim.zero_grad()
            logits, feats = self.model(inputs)
            losses = {'loss/ce': self.crit(logits, targets)}
            if self.flags['sub_batches'] > 1:
                losses['loss/top'] = self._topological_crit(feats)
            if self.global_h0_regularizer is not None:
                losses[
                    'loss/global_top'] = self.global_h0_regularizer.criterium(
                        idcs, targets, feats)

            # loss
            for k, l in losses.items():
                losses_means[k] += l.item()
            weight += targets.size(0) / self.flags['batch_size']
            loss = sum(losses.values())

            loss.backward()
            self.optim.step()

            # accuracy
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        self.scheduler.step()

        for k in losses_means:
            losses_means[k] /= weight

        return {
            'acc': 100.0 * correct / total,
            'loss_': sum(losses_means.values()),
            **losses_means
        }

    def _run_batches_valid(self, set_name):
        self.model.eval()
        correct, total = 0, 0
        losses_means, weight = defaultdict(float), 0
        with torch.no_grad():
            for idcs, inputs, targets in self.loaders[set_name]:
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                logits, feats = self.model(inputs)
                losses = {'loss/ce': self.crit(logits, targets)}
                if self.flags['sub_batches'] > 1:
                    losses['loss/top'] = self._topological_crit(feats)
                # if self.global_h0_regularizer is not None:
                #     losses['loss/global_top'] = self.global_h0_regularizer.criterium(idcs, targets, feats)

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

        return {
            'acc': 100.0 * correct / total,
            'loss_': sum(losses_means.values()),
            **losses_means
        }

    def pers_l2(self, feats):
        # from torchph.pershom import vr_persistence
        n = feats.size(0)
        x = feats.unsqueeze(1).expand(
            [feats.size(0), feats.size(0),
             feats.size(1)])
        x = x.transpose(0, 1) - x
        x = x.norm(dim=2, p=2)
        # return vr_persistence(x, 0, 0)[0][0][:, 1]
        mask = torch.eye(n, dtype=torch.bool).logical_not()
        result = x[mask].view(n, n - 1).min(axis=1).values
        mask = torch.ones_like(result, dtype=torch.bool)
        mask[result.argmin()] = False
        return result[mask]

    def _topological_crit(self, feats):
        # topological loss
        if self.flags['h0_decay'] == 0.0:
            return torch.tensor(0.0).to(self.dev)

        top_scale = self.flags['h0_dens']
        actual_batch_size = feats.size(0)
        n = self.flags['sub_batches']
        top_loss = torch.tensor(0.0).to(self.dev)
        for i in range(actual_batch_size // n):
            z_sample = feats[i * n:(i + 1) * n, :].contiguous()
            lt = self.pers_l2(z_sample)

            top_loss += (lt - top_scale).abs().sum()
        top_loss /= float(actual_batch_size // n)

        return self.flags['h0_decay'] * top_loss
