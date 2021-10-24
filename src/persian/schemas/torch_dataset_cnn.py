from persian.schemas.torch_dataset import DatasetTorchSchema
from persian.errors.flags_incompatible import IncompatibleFlagsError
from persian.errors.value_flag_unknown import UnknownFlagValue

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR


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
                self.model.parameters(),
                lr=self.flags['lr'],
                weight_decay=self.flags['w_decay'],
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise UnknownFlagValue(f"Unknown value of `optim`")

        if self.flags['sched'] == 'None':
            self.scheduler = StepLR(self.optim, step_size=1, gamma=1.0)
        elif self.flags['sched'] == 'cos':
            self.scheduler = CosineAnnealingLR(self.optim,
                                               T_max=self.flags['epochs'],
                                               eta_min=0,
                                               last_epoch=-1)
        else:
            assert False

        self.crit = nn.CrossEntropyLoss()

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        if set_name == 'TRAIN':
            self.metrics[set_name] = self._run_batches_train(set_name)
        else:
            self.metrics[set_name] = self._run_batches_valid(set_name)

    def run_inference(self, input):
        return self.model([input])[0]

    def _run_batches_train(self, set_name):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in self.loaders[set_name]:
            inputs, targets = inputs.to(self.dev), targets.to(self.dev)
            self.optim.zero_grad()
            logits, feats = self.model(inputs)
            loss = self.crit(logits, targets)
            loss += self._topological_crit(feats)

            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return dict(loss=train_loss / total, acc=100. * correct / total)

    def _run_batches_valid(self, set_name):
        self.model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.loaders[set_name]:
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                logits, feats = self.model(inputs)
                loss = self.crit(logits, targets)
                loss += self._topological_crit(feats)

                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return dict(loss=test_loss / total, acc=100. * correct / total)

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

        top_scale = 0.7
        actual_batch_size = feats.size(0)
        n = self.flags['sub_batches']
        top_loss = torch.tensor(0.0).to(self.dev)
        for i in range(actual_batch_size // n):
            z_sample = feats[i * n:(i + 1) * n, :].contiguous()
            lt = self.pers_l2(z_sample)

            top_loss += (lt - top_scale).abs().sum()
        top_loss /= float(actual_batch_size // n)

        return self.flags['h0_decay'] * top_loss
