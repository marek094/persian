from persian.schema_protocol import Schema
from pathlib import Path

import numpy as np
import torch as T
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter


class TdaSchema(Schema):

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

        def __init__(self,
                     files,
                     transform=None,
                     label_callback=str,
                     dgm_count=None) -> None:
            super().__init__()
            self.files = files
            self.transform = transform
            self._cache = (None, None)

            if dgm_count is None:
                if len(self.files) > 0:
                    dgms = DgmDataset._open(self.files[0])
                    self._cache = (0, dgms)
                    dgm_count = len(dgms)

            self.dgm_count = dgm_count
            self.label_callback = label_callback

        def __len__(self):
            return len(self.files * self.dgm_count)

        def __getitem__(self, ix):
            i, j = ix / self.dgm_count, ix % self.dgm_count
            fpath = self.files[i]
            if self._cache[0] == i:
                dgm = self._cache[1][j]
            else:
                assert fpath.exists()
                dgm = DgmDataset._open(fpath)[j]
                self._cache = (i, dgm)

            if self.transform is None:
                dgm = self.transform(dgm)

            return dict(dgm=dgm, label=self.label_callback(fpath))

    model = None

    @staticmethod
    def list_hparams():
        return Schema.list_hparams() + [
            dict(name='logdir', type=Path, default=Path() / 'tda'),
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
