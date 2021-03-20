from torchvision.transforms import transforms
from persian.schema_protocol import Schema
from pathlib import Path

import numpy as np
import torch as T
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class TdaSchema(Schema):

    @staticmethod
    def dgm_from_gtda(dgm_gtda):
        assert len(dgm_gtda.shape) == 2
        dims = dgm_gtda[:, 2]
        dims = np.unique(np.sort(dims))
        return {dim: dgm_gtda[dgm_gtda[:, 2] == dim][:, :2] for dim in dims}

    class DgmDataset(Dataset):
        """
        expect all files has same number of diagrams
        """

        @staticmethod
        def _open(fpath: Path):
            assert fpath.exists()
            if fpath.suffix == '.npz':
                return np.load(fpath)['dgms'].astype(np.float32)
            raise NotImplementedError(
                f'Format {fpath.suffix} is not supported yet')

        def __init__(self,
                     data,
                     transform=None,
                     dgm_count=None,
                     load_all=False,
                     from_gtda=True) -> None:
            super().__init__()
            self.files, self.labels = zip(*data)
            self.transform = transform
            self._cache = (None, None)

            # determine shape by the first
            if dgm_count is None:
                if len(self.files) > 0:
                    dgms = TdaSchema.DgmDataset._open(self.files[0])
                    self._cache = (0, dgms)
                    dgm_count = len(dgms)
                else:
                    dgm_count = 0

            self.dgm_count = dgm_count
            self.from_gtda = from_gtda
            self.load_all = load_all

            if self.load_all:
                self.mem = [None] * len(self)
            else:
                self.mem = None

        def __len__(self):
            return len(self.files) * self.dgm_count

        def getindex(self, i, j):
            fpath = self.files[i]

            if self._cache[0] == i:
                dgm = self._cache[1][j]
            else:
                assert fpath.exists()
                dgms = TdaSchema.DgmDataset._open(fpath)
                self._cache = (i, dgms)
                dgm = dgms[j]

            if self.from_gtda:
                dgm = TdaSchema.dgm_from_gtda(dgm)

                if isinstance(self.transform, list):
                    dgm = [{dim: trfm(d)
                            for dim, d in dgm.items()}
                           for trfm in self.transform]
                elif self.transform is not None:
                    dgm = {dim: self.transform(d) for dim, d in dgm.items()}
            else:
                if isinstance(self.transform, list):
                    dgm = [trfm(dgm) for trfm in self.transform]
                elif self.transform is not None:
                    dgm = self.transform(dgm)

            return (dgm, self.labels[i])

        def __getitem__(self, ix):
            i, j = divmod(ix, self.dgm_count)

            if not self.load_all:
                return self.getindex(i, j)

            self.mem[ix] = (self.mem[ix] or self.getindex(i, j))
            return self.mem[ix]

    model = None

    @staticmethod
    def list_hparams():
        return Schema.list_hparams() + [
            dict(name='logdir', type=Path, default=Path() / 'tda'),
            dict(name='seed', type=int, default=0)
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags)

        logs = self.flags['logdir'] / f'{self.as_hstr()}_'
        self._writer = SummaryWriter(logs)
        self._writer_epoch = 0

        T.set_deterministic(True)
        T.manual_seed(self.flags['seed'])
        self.dev = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.loaders = {}

    def update_infoboard(self) -> None:
        for phase, metric_set in self.metrics.items():
            for name, value in metric_set.items():
                self._writer.add_scalar(f'{name}/{phase}', value,
                                        self._writer_epoch)
        self._writer_epoch += 1

    def _placed_loader(self, set_name):
        return (
            (i.to(self.dev), o.to(self.dev)) for i, o in self.loaders[set_name])
