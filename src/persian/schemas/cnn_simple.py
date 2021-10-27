from torch._C import default_generator
from torch.nn.modules import dropout, padding, pooling
from persian.errors import UnknownFlagValueError, IncompatibleFlagsError
from persian.schemas.torch_dataset_cnn import CnnDatasetTorchSchema

from torch import nn


class SimpleCnnSchema(CnnDatasetTorchSchema):
    @staticmethod
    def list_hparams():
        return CnnDatasetTorchSchema.list_hparams() + [
            dict(name='nlayers', type=int, default=5, choices=[5, 13]),
            dict(name='width', type=int, default=128),
            dict(name='symetric', type=bool, default=True),
            dict(name='dropout', type=float, default=0.0),
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags=flags)

    def make_cnn(self):
        assert len(self.loaders) > 0
        dataset = next(iter(self.loaders.values())).dataset
        passed_flags = ['nlayers', 'width', 'symetric', 'dropout']
        return Model(nclasses=len(dataset.classes),
                     batch_norm=True,
                     cls_spectral_norm=False,
                     **{k: self.flags[k]
                        for k in passed_flags})


class Model(nn.Module):
    def __init__(self, nclasses, symetric, width, nlayers, dropout, batch_norm,
                 cls_spectral_norm):
        super().__init__()

        def conv_block(cin,
                       cout,
                       padding=1,
                       kernel=3,
                       pooling=2,
                       leaky=None,
                       dropout=0.0):
            return [
                nn.Conv2d(cin,
                          cout,
                          kernel_size=kernel,
                          padding=padding,
                          stride=1),
                nn.BatchNorm2d(cout) if batch_norm else nn.Identity(),
                nn.LeakyReLU(leaky) if leaky is not None else nn.ReLU(),
                nn.MaxPool2d(pooling, padding=0)
                if pooling is not None else nn.Identity(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            ]

        if nlayers == 5:
            if symetric:
                # yapf: disable
                feat_ext_layers = (
                    conv_block(      3, width*1) +
                    conv_block(width*1, width*2) +
                    conv_block(width*2, width*4) +
                    conv_block(width*4, width*2) +
                    conv_block(width*2, width*1) +
                    [nn.Flatten()]
                )
                # yapf: enable
                linear_in = width
            else:
                # yapf: disable
                feat_ext_layers = (
                    conv_block(      3, width*1, pooling=None, leaky=0.01) +
                    conv_block(width*1, width*2, leaky=0.01) +
                    conv_block(width*2, width*4, leaky=0.01) +
                    conv_block(width*4, width*8, leaky=0.01) +
                    conv_block(width*8, width*16,leaky=0.01) +
                    [
                        nn.MaxPool2d(4),
                        nn.Flatten()
                    ]
                )
                # yapf: enable
                linear_in = 10

        elif nlayers == 13:
            if not symetric:
                raise IncompatibleFlagsError(
                    "Non-symetrical version is not implemented")

            # this is SimpleCNN13
            # yapf: disable
            feat_ext_layers = (
                conv_block(    3, width, pooling=None, leaky=0.1) +
                conv_block(width, width, pooling=None, leaky=0.1) +
                conv_block(width, width, dropout=dropout, leaky=0.1) +

                conv_block(width*1, width*2, pooling=None, leaky=0.1) +
                conv_block(width*2, width*2, pooling=None, leaky=0.1) +
                conv_block(width*2, width*2, dropout=dropout, leaky=0.1) +

                conv_block(width*2, width*4, pooling=None, padding=0, leaky=0.1) +
                conv_block(width*4, width*2, kernel=1, pooling=None, padding=0, leaky=0.1) +
                conv_block(width*2, width*1, kernel=1, pooling=None, padding=0, leaky=0.1) +
                [
                    nn.AvgPool2d(6, stride=2, padding=0),
                    nn.Flatten()
                ]
            )
            # yapf: enable
            linear_in = width

        else:
            raise UnknownFlagValueError("Undefined number of layers `nlayers`")

        self.feat_ext = nn.Sequential(*feat_ext_layers)
        cls = nn.Linear(linear_in, nclasses)

        if cls_spectral_norm:
            nn.utils.spectral_norm(cls)

        self.cls = nn.Sequential(cls)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z
