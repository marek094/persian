from torch.utils.data.dataset import Subset
from persian.schemas.torch import TorchSchema
from persian.errors import *

import torch.utils.data as data
import torch as T

from pathlib import Path
import pandas as pd
import numpy as np
from collections import OrderedDict


class ZoosetTorchSchema(TorchSchema):
    @staticmethod
    def list_hparams():
        return TorchSchema.list_hparams() + [
            dict(name='batch_size', type=int, default=32),
            dict(name='target', type=str, default='test_accuracy'),
            dict(name='split_seed', type=int, default=2021),
            dict(name='train_size', type=int, default=0)
        ]

    def __init__(self, flags={}):
        super().__init__(flags)

        self.unsplitted_dataset = None
        self.split_indices = None
        self.loaders = {}

    def _init_dataset_if_not_present(self):
        if self.unsplitted_dataset is not None:
            return

        assert self.split_indices is None

        self.unsplitted_dataset = ZoosetDataset(
            target_name=self.flags['target'],
            dev=self.dev,
        )

        np.random.seed(self.flags['split_seed'])
        n = len(self.unsplitted_dataset)
        half = round(n / 2)

        train_size = self.flags[
            'train_size'] if self.flags['train_size'] > 0 else half
        perm = np.random.permutation(n)
        self.split_indices = {
            'TRAIN': perm[0:train_size],
            'VALID': perm[half:n],
        }

    def prepare_dataset(self, set_name):
        self._init_dataset_if_not_present()

        assert set_name in ['TRAIN', 'VALID']
        dataset = Subset(
            self.unsplitted_dataset,
            indices=self.split_indices[set_name],
        )

        self.loaders[set_name] = data.DataLoader(
            dataset=dataset,
            batch_size=self.flags['batch_size'],
            shuffle=set_name == 'TRAIN',
            collate_fn=ZoosetDataset.collate_fn,
        )


class ZoosetDataset(data.Dataset):
    @staticmethod
    def collate_fn(batch):
        nets = [net for net, _ in batch]
        tgts = [[tgt] for _, tgt in batch]
        return nets, T.tensor(tgts)

    def __init__(self, target_name, dev):
        self.data_folder = Path() / '../data'
        assert self.data_folder.exists()
        self.metrics_path = self.data_folder / 'metrics.csv'
        assert self.metrics_path.exists()
        self.weights_path = self.data_folder / 'weights_step86.npz'
        assert self.weights_path.exists()

        self.target_name = target_name
        self.dev = dev

        self.metrics = None
        self.weights = None
        self.cache = None

    def _load_data_if_not_present(self):
        if self.metrics is not None or self.weights is not None:
            assert self.metrics is not None
            assert self.weights is not None
            return

        metrics = pd.read_csv(self.metrics_path)

        if self.weights_path.suffix == '.npz':
            weights = np.load(self.weights_path)['arr_0']
            metrics = metrics[metrics.step == 86]
        else:
            weights = np.load(self.weights_path)

        self.weights = weights
        self.metrics = metrics
        self.cache = [None] * len(metrics.index)
        if self.target_name not in self.metrics.columns:
            raise UnknownFlagValueError(
                f'Target {self.target_name} does not exists in the loaded dataframe'
            )
        col = self.metrics[self.target_name]
        self.targets = np.array(col).astype(np.float32)

    def __len__(self):
        self._load_data_if_not_present()
        return len(self.metrics.index)

    def __getitem__(self, index):
        self._load_data_if_not_present()

        if self.cache[index] is not None:
            return self.cache[index], self.targets[index]

        w = self.weights[index]
        act = self.metrics['config.activation'].iloc[index]
        step = self.metrics.step.iloc[index]

        zoomodel = TorchCNN(act, use_last_layer=False).set_weight(w)
        zoomodel = zoomodel.to(self.dev)
        assert step == 86

        self.cache[index] = zoomodel
        return self.cache[index], self.targets[index]


class TorchCNN(T.nn.Sequential):
    def __init__(self, activation, use_softmax=False, use_last_layer=True):
        if activation == 'relu':
            Act = T.nn.ReLU
        elif activation == 'tanh':
            Act = T.nn.Tanh
        else:
            assert False, f"Activation '{activation}' unknown"

        # yapf: disable
        ordered_dict = [
            ('conv_0',
                T.nn.Conv2d( 1, 16, 3, 2)),
            ('act_0',
                Act()),

            ('conv_1',
                T.nn.Conv2d(16, 16, 3, 2)),
            ('act_1',
                Act()),

            ('conv_2',
                T.nn.Conv2d(16, 16, 3, 2)),
            ('act_2',
                Act()),

            ('pool',
                T.nn.AdaptiveAvgPool2d(1)),
            ('flatten',
                T.nn.Flatten()),
            ('dense',
                T.nn.Linear(16, 10)),

        #             ('softmax',
        #                 T.nn.Softmax(dim=1))
        ]
        # yapf: enable

        super().__init__(OrderedDict(ordered_dict))
        self.use_last_layer = use_last_layer

    @staticmethod
    def _set_conv(conv, bias, kernel):
        conv.bias = T.nn.parameter.Parameter(
            data=T.tensor(bias),
            requires_grad=False,
        )
        conv.weight = T.nn.parameter.Parameter(
            data=T.tensor(kernel),
            requires_grad=False,
        )

    @staticmethod
    def _set_dense(dense, bias, kernel):
        dense.bias = T.nn.parameter.Parameter(
            data=T.tensor(bias),
            requires_grad=False,
        )

        dense.weight = T.nn.parameter.Parameter(
            data=T.tensor(kernel),
            requires_grad=False,
        )

    @staticmethod
    def _shape_weight(weight):
        # yapf: disable
        table = [
          (0,    16,   (16,), (0,)), (16,   160,  (3, 3,  1, 16), (3,2,0,1)),
          (160,  176,  (16,), (0,)), (176,  2480, (3, 3, 16, 16), (3,2,0,1)),
          (2480, 2496, (16,), (0,)), (2496, 4800, (3, 3, 16, 16), (3,2,0,1)),
          (4800, 4810, (10,), (0,)), (4810, 4970, (16, 10),       (1,0)),
        ]
        # yapf: enable
        return [
            weight[ixb:ixe].reshape(shape).transpose(transp)
            for ixb, ixe, shape, transp in table
        ]

    def set_weight(self, weight):
        l = dict(self.named_children())
        sw = self._shape_weight(weight)
        # yapf: disable
        if 'conv_0' in l:
            self._set_conv(l['conv_0'], sw[0], sw[1])
        if 'conv_1' in l:
            self._set_conv(l['conv_1'], sw[2], sw[3])
        if 'conv_2' in l:
            self._set_conv(l['conv_2'], sw[4], sw[5])
        if 'dense' in l:
            self._set_dense(l['dense'], sw[6], sw[7])
        # yapf: enable
        return self
