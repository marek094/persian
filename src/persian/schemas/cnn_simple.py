from persian.schemas.torch_dataset_cnn import CnnDatasetTorchSchema

from torch import nn


class SimpleCnnSchema(CnnDatasetTorchSchema):

    @staticmethod
    def list_hparams():
        return CnnDatasetTorchSchema.list_hparams() + [
            dict(name='nlayers', type=int, default=5, choices=[5, 13]),
            dict(name='width', type=int, default=128),
            dict(name='symetric', type=bool, default=True),
        ]

    def __init__(self, flags={}) -> None:
        super().__init__(flags=flags)

    def make_cnn(self):
        assert len(self.loaders) > 0
        dataset = next(iter(self.loaders.values())).dataset
        return Model(
            num_classes=len(dataset.classes),
            symetric=self.flags['symetric'],
            width=self.flags['width'],
            nlayers=self.flags['nlayers'],
            batch_norm=True,
            cls_spectral_norm=False,
        )

class Model(nn.Module):
    def __init__(self,
                num_classes,
                symetric,
                width,
                nlayers,
                batch_norm,
                cls_spectral_norm):
        super().__init__()

        with_leaky = symetric

        def conv_block(cin, cout, with_pooling=True):
            return [
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(cout) if batch_norm else nn.Identity(),
                nn.ReLU() if not with_leaky else nn.LeakyReLU(),
                nn.MaxPool2d(2) if with_pooling else nn.Identity(),
            ]
        
        if nlayers == 5:
            if symetric:
                feat_ext_layers = (
                    conv_block(3    *1, width*1) +
                    conv_block(width*1, width*2) + 
                    conv_block(width*2, width*4) +
                    conv_block(width*4, width*2) +
                    conv_block(width*2, width*1) +
                    [nn.Flatten()]
                )
                linear_in = width
            else:
                feat_ext_layers = (
                    conv_block(3    *1, width*1, with_pooling=False) +
                    conv_block(width*1, width*2) +
                    conv_block(width*2, width*4) +
                    conv_block(width*4, width*8) +
                    conv_block(width*8, width*16) +
                    [
                        nn.MaxPool2d(4), 
                        nn.Flatten()
                    ]
                )
                linear_in = 10

        elif nlayers == 13:
            feat_ext_layers = [nn.Flatten()]

        else:
            assert False


        self.feat_ext = nn.Sequential(*feat_ext_layers)
        cls = nn.Linear(linear_in, num_classes)

        if cls_spectral_norm:
            nn.utils.spectral_norm(cls)

        self.cls = nn.Sequential(cls)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z

