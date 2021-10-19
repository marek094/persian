from persian.schemas.torch import TorchSchema

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import copy
from numpy.random import shuffle


class DatasetTorchSchema(TorchSchema):

    @staticmethod
    def list_hparams():
        return TorchSchema.list_hparams() + [
            dict(name='batch_size', type=int, default=128),
            dict(name='noise', type=int, default=0, range=(0, 30, 10)),
            dict(name='dataset', type=str, default='cifar10'),
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags)
        self.loaders = {}

    @staticmethod
    def _get_dataset_label_noise(trainset, noise_size):
        ######## shuffle
        train_size = len(trainset)
        shuffle_targets_set = [
            copy.deepcopy(trainset.targets[idx])
            for idx in range(train_size - noise_size, train_size)
        ]
        shuffle(shuffle_targets_set)
        for idx in range(train_size - noise_size, train_size):
            trainset.targets[idx] = shuffle_targets_set[idx - train_size +
                                                        noise_size]
        return trainset

    def _dataset_from_name(self, name):
        dsets = {
            'cifar10'  : datasets.CIFAR10,
            'cifar100' : datasets.CIFAR100,
            'mnist'    : datasets.MNIST,
            'fmnist'   : datasets.FashionMNIST,
            'svhn'     : datasets.SVHN,
        }

        if name in dsets:
            return dsets[name]
        return None

    def prepare_dataset(self, set_name):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        is_train = set_name == 'TRAIN'
        dataset_cls = self._dataset_from_name(self.flags['dataset'])
        self.dataset_meta = dataset_cls.meta
        ds = dataset_cls(root='../data',
                              train=is_train,
                              download=True,
                              transform=transform)
        if is_train:
            noise_size = len(ds) * self.flags['noise'] // 100
            ds = self._get_dataset_label_noise(ds, noise_size=noise_size)
        bs = self.flags['batch_size']
        self.loaders[set_name] = DataLoader(ds, batch_size=bs, shuffle=is_train)