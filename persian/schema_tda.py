from persian.schema_protocol import Schema
from pathlib import Path

import numpy as np
import torch as T
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class TdaSchema(Schema):

    @staticmethod
    def dgm_from_gtda(dgm_gtda):
        # if len(dgm_gtda.shape) < 2:
        #     print('NO')
        #     return {
        #         0: np.zeros((0, 2)),
        #         1: np.zeros((0, 2)),
        #         2: np.zeros((0, 2)),
        #     }
        # else:
        #     print('OK')
        assert len(dgm_gtda.shape) == 2
        dims = dgm_gtda[:, 2]
        dims = np.unique(np.sort(dims))
        return {dim: dgm_gtda[dgm_gtda[:, 2] == dim][:, :2] for dim in dims}

    class DgmDataset(Dataset):
        """
        expect all files has same number of diagrams
        """

        @staticmethod
        def _open(fpath):
            assert fpath.exists()
            if fpath.suffix == '.npz':
                return np.load(fpath)['dgms']
            raise NotImplementedError(
                f'Format {fpath.suffix} is not supported yet')

        def __init__(self, data, transform=None, dgm_count=None) -> None:
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

        def __len__(self):
            return len(self.files) * self.dgm_count

        def __getitem__(self, ix):
            i, j = ix // self.dgm_count, ix % self.dgm_count
            fpath = self.files[i]
            if self._cache[0] == i:
                dgm = self._cache[1][j]
            else:
                assert fpath.exists()
                dgms = TdaSchema.DgmDataset._open(fpath)
                self._cache = (i, dgms)
                dgm = dgms[j]

            dgm = TdaSchema.dgm_from_gtda(dgm)

            if self.transform is None:
                dgm = {dim: self.transform(d) for dim, d in dgm.items()}

            return (dgm, self.labels[i])

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

    def update_infoboard(self) -> None:
        for phase, metric_set in self.metrics.items():
            for name, value in metric_set.items():
                self._writer.add_scalar(f'{name}/{phase}', value,
                                        self._writer_epoch)
        self._writer_epoch += 1
