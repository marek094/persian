
import torch as T
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from persian.schemas.torch_dataset import DatasetTorchSchema

class CnnDatasetTorchSchema(DatasetTorchSchema):

    @staticmethod
    def list_hparams():
        return DatasetTorchSchema.list_hparams() + [
            dict(name='epochs', type=int, default=80),
            dict(name='lr', type=float, default=0.1, range=(0.01, 0.21, 0.01)),
            dict(name='weight_decay', type=float, default=5e-4),
            dict(name='lr_decay', type=int, default=1000),
            # dict(name='width', type=int, default=10, range=(2, 80, 4)),
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags)

    def make_cnn(self):
        raise NotImplementedError()

    def prepare_model(self):
        self.model = self.make_cnn()
        self.model = self.model.to(self.dev)

    def prepare_criterium(self):
        self.optim = T.optim.Adam(self.model.parameters(),
                                  lr=self.flags['lr'],
                                  weight_decay=self.flags['weight_decay'])

        self.scheduler = StepLR(self.optim,
                                step_size=self.flags['lr_decay'],
                                gamma=0.1)

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
        with T.no_grad():
            for inputs, targets in self.loaders[set_name]:
                inputs, targets = inputs.to(self.dev), targets.to(self.dev)
                logits, feats = self.model(inputs)
                loss = self.crit(logits, targets)

                test_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return dict(loss=test_loss / total, acc=100. * correct / total)
