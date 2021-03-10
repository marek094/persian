from torchvision.transforms import transforms
from torchvision.transforms.transforms import Lambda
from persian.schema_tda import TdaSchema
from persian.layer_silhouette import SilhouetteLayer

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import ToTensor, Lambda
from torch.utils.data import DataLoader

from pathlib import Path


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class Expt2Schema(TdaSchema):

    @staticmethod
    def list_hparams():
        return TdaSchema.list_hparams() + [
            dict(name='datadir', type=Path, default='../data'),
            dict(name='TRAIN_sd', type=int, default=100),
            dict(name='VALID_sd', type=int, default=101),
            dict(name='batch_size', type=int, default=8),
            dict(name='epochs', type=int, default=40),
            dict(name='lr', type=float, default=0.001),
            dict(name='gamma', type=float, default=0.985),
            dict(name='power_init', type=float, default=1.),
            dict(name='lo', type=float, default=0),
            dict(name='hi', type=float, default=0.2),
            dict(name='bins', type=int, default=64),
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags)
        self.loaders = {}

    def prepare_dataset(self, set_name):
        sd = self.flags[f'{set_name}_sd']

        files = self.flags['datadir'].glob(f'*[,_]s{sd}[,_]*[,_]i32[,_]*.npz')
        ds = TdaSchema.DgmDataset(
            files=list(files),
            transform=ToTensor(),
            label_callback=lambda p: [
                T.LongTensor([0 if float(x[1:]) == 0.05 else 1])
                for x in p.stem.split('_')[1].split(',')
                if x[0] == 'r'
            ][0])

        def collate(batch_list):
            dims = (0, 1, 2)
            labels = T.cat([label for _, label in batch_list])

            dgms = {}
            for dim in dims:
                max_pts = max(dgm[dim].shape[0] for dgm, _ in batch_list)

                #  padding to shape (batch, max_pts, 2)
                for dgm, _ in batch_list:
                    n_pts = dgm[dim].shape[0]
                    tmp = T.zeros(max_pts, 2)
                    tmp[:n_pts] = T.Tensor(dgm[dim])
                    dgm[dim] = tmp
                    # print(dim, dgm[dim].shape)

                dgms[dim] = T.stack([dgm[dim] for dgm, _ in batch_list])

            return dgms, labels

        bs = self.flags['batch_size']
        is_train = set_name == 'TRAIN'
        self.loaders[set_name] = DataLoader(ds,
                                            batch_size=bs,
                                            collate_fn=collate,
                                            shuffle=is_train)

    def prepare_model(self):
        W = self.flags['bins']
        model = nn.Sequential(
            SilhouetteLayer(
                init_power=self.flags['power_init'],
                n_bins=self.flags['bins'],
                lo=self.flags['lo'],
                hi=self.flags['hi'],
            ), nn.Conv1d(3, 8, kernel_size=9, stride=4), nn.BatchNorm1d(8),
            nn.ReLU(), nn.Conv1d(8, 16, kernel_size=5, stride=2),
            nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2), nn.BatchNorm1d(32),
            nn.ReLU(), nn.Flatten(), nn.Linear(W // 2, 2))

        self.model = model.to(self.dev)

    def prepare_criterium(self):
        self.optim = T.optim.SGD(
            self.model.parameters(),
            lr=self.flags['lr'],
            momentum=0.9,
            weight_decay=0.,
        )

        self.sched = T.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=1,
            gamma=self.flags['gamma'],
        )

        self.crit = nn.CrossEntropyLoss().to(self.dev)

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        if set_name == 'TRAIN':
            self.metrics[set_name] = self._run_batches_train(set_name)
        else:
            self.metrics[set_name] = self._run_batches_valid(set_name)

    def _run_batches_train(self, set_name):
        self.model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in self.loaders[set_name]:
            inputs = {dim: inp.to(self.dev) for dim, inp in inputs.items()}
            targets = targets.to(self.dev)

            self.optim.zero_grad()
            outputs = self.model(inputs)
            loss = self.crit(outputs, targets)
            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        self.sched.step()
        return dict(
            loss=train_loss / total,
            acc=100. * correct / total,
        )

    def _run_batches_valid(self, set_name):
        self.model.eval()
        valid_loss, correct, total = 0, 0, 0
        with T.no_grad():
            for inputs, targets in self.loaders[set_name]:
                inputs = {dim: inp.to(self.dev) for dim, inp in inputs.items()}
                targets = targets.to(self.dev)

                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)

                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return dict(
            loss=valid_loss / total,
            acc=100. * correct / total,
        )


if __name__ == "__main__":
    expt = Expt2Schema()
