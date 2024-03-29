from persian.schemas.schema import Schema
import persian.config as config

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch


class TorchSchema(Schema):

    model = None

    @staticmethod
    def list_hparams():
        return Schema.list_hparams() + [
            dict(name='logdir', type=Path, default=Path() / 'runs'),
            dict(name='seed', type=int, default=2022),
            dict(name='ndalg', type=bool, default=True),
        ]

    def __init__(self, flags={}):
        super().__init__(flags)
        # writer
        logs = self.flags['logdir'] / f'{self.as_hstr()}_'
        self.logs_path = logs
        self._writer = SummaryWriter(logs)
        self._writer_epoch = 0
        # deterministic mode
        if self.flags['ndalg']:
            torch.use_deterministic_algorithms(True)

        if config.AUTOTUNER_TURN_ON:
            torch.backends.cudnn.benchmark = True

        torch.manual_seed(self.flags['seed'])
        # device
        self.dev = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

    def update_infoboard(self) -> None:
        for phase, metric_set in self.metrics.items():
            for name, value in metric_set.items():
                self._writer.add_scalar(f'{name}/{phase}', value,
                                        self._writer_epoch)
        self._writer_epoch += 1

    def pack_model_params(self):
        if self.model is None:
            return None
        return self.model.state_dict()

    def load_model_params(self, state_dict):
        if self.model is None:
            return None
        self.model.load_state_dict(state_dict)
