from persian.schemas.torch import TorchSchema

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler, BatchSampler
import torch

from numpy.random import shuffle
from typing import List, Iterator
import copy


class DatasetTorchSchema(TorchSchema):
    @staticmethod
    def list_hparams():
        return TorchSchema.list_hparams() + [
            dict(name='batch_size', type=int, default=128),
            dict(name='noise', type=int, default=0, range=(0, 30, 10)),
            dict(name='dataset', type=str, default='cifar10'),
            dict(name='train_size', type=int, default=None),
            dict(name='sub_batches', type=int, default=None),
            dict(name='aug', type=str, default=None),
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
            'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'mnist': datasets.MNIST,
            'fmnist': datasets.FashionMNIST,
            'svhn': datasets.SVHN,
        }

        if name in dsets:
            return dsets[name]
        return None

    def _augmentation_from_name(self, name):
        aug_transforms = [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]

        if name is None:
            pass
        elif name == 'tdd':
            aug_transforms += [
                transforms.Pad([2, 2, 2, 2]),
                transforms.RandomCrop([32, 32]),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        else:
            assert False

        return aug_transforms

    def prepare_dataset(self, set_name):

        is_train = set_name == 'TRAIN'
        dataset_cls = self._dataset_from_name(self.flags['dataset'])
        aug_transforms = self._augmentation_from_name(self.flags['aug'])
        self.dataset_meta = dataset_cls.meta
        ds = dataset_cls(root='../data',
                         train=is_train,
                         download=True,
                         transform=transforms.Compose(aug_transforms))

        shuffle = False
        sampler = None
        batch_sampler = None
        batch_size = self.flags['batch_size']

        if is_train:
            noise_size = round(len(ds) * self.flags['noise'] / 100)
            ds = self._get_dataset_label_noise(ds, noise_size=noise_size)
            shuffle = True
            size_limit = self.flags['train_size']

            if self.flags['sub_batches'] is not None:
                sub_batch_sampler = HomogeneousRandomBatchSampler(
                    dataset=ds,
                    batch_size=self.flags['sub_batches'],
                    drop_last=True,
                    size_limit=size_limit,
                )
                n_sub_batches = self.flags['batch_size'] // self.flags[
                    'sub_batches']
                batch_sampler = ConcatenatingBatchSampler(
                    sub_batch_sampler,
                    batch_size=n_sub_batches,
                    drop_last=False)

                batch_size = 1  # default for DataLoader
                shuffle = None
            elif size_limit is not None:
                sampler = SubsetRandomSampler(range(size_limit))
                shuffle = None

        self.loaders[set_name] = DataLoader(ds,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            sampler=sampler,
                                            batch_sampler=batch_sampler)


class HomogeneousRandomBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size, drop_last, size_limit=None):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.size_limit = len(dataset) if size_limit is None else size_limit
        self._build_generator()
        self._limit_per_targets(dataset)

    def _build_generator(self):
        generator = torch.Generator()
        generator.manual_seed(
            int(torch.empty((), dtype=torch.int64).random_().item()))
        self.generator = generator

    def _limit_per_targets(self, dataset):
        per_class_samplers = []
        q, reminder = divmod(self.size_limit, len(dataset.class_to_idx))
        for class_idx in dataset.class_to_idx.values():
            idcs = torch.where(torch.tensor(dataset.targets) == class_idx)[0]
            perm = torch.randperm(len(idcs), generator=self.generator)
            q2 = q
            if reminder > 0:
                q2 += 1
                reminder -= 1
            sampler = SubsetRandomSampler(idcs[perm][:q2],
                                          generator=self.generator)
            batch_sampler = BatchSampler(sampler,
                                         batch_size=self.batch_size,
                                         drop_last=self.drop_last)
            per_class_samplers.append(batch_sampler)

        self.per_class_samplers = per_class_samplers
        self.len = self.size_limit

    def _get_next_values(self):
        next_values = []
        for sampler in self.per_class_samplers:
            it = iter(sampler)
            next_values.append((it, next(it, None)))
        return next_values

    def __iter__(self) -> Iterator[List[int]]:
        next_values = self._get_next_values()
        while True:
            active_nexts = [
                cls for cls, (_, nv) in enumerate(next_values)
                if nv is not None
            ]
            if len(active_nexts) == 0:
                break
            idx = torch.randint(0,
                                len(active_nexts), [],
                                generator=self.generator)
            cls = active_nexts[idx]
            it, batch = next_values[cls]
            next_values[cls] = (it, next(it, None))
            yield batch

    def __len__(self) -> int:
        return self.len


class ConcatenatingBatchSampler(BatchSampler):
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)

    def __iter__(self) -> Iterator[List[int]]:
        for batch in super().__iter__():
            yield [idx for subbatch in batch for idx in subbatch]
