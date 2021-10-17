from persian.schemas.torch_dataset_cnn import CnnDatasetTorchSchema

from torch import nn


class SimpleCnnSchema(CnnDatasetTorchSchema):

    @staticmethod
    def list_hparams():
        return CnnDatasetTorchSchema.list_hparams()

    def make_cnn(self):
        return Model(num_classes=10, batch_norm=True, cls_spectral_norm=None)

    def __init__(self, flags={}) -> None:
        super().__init__(flags=flags)

class Model(nn.Module):
    def __init__(self,
                num_classes,
                batch_norm,
                cls_spectral_norm):
        super().__init__()

        def conv_block(cin, cout):
            res = []
            res.append(nn.Conv2d(cin, cout, 3, padding=1))
            if batch_norm:
                res.append(nn.BatchNorm2d(cout))
            res.append(nn.LeakyReLU())
            res.append(nn.MaxPool2d(2, stride=2, padding=0))
            return res

        feat_ext_layers = (
            conv_block(3, 128) +
            conv_block(128, 256) + 
            conv_block(256, 512) +
            conv_block(512, 256) +
            conv_block(256, 128) +
            [nn.Flatten()]
        )
        self.feat_ext = nn.Sequential(*feat_ext_layers)

        cls = nn.Linear(128, num_classes)
        if cls_spectral_norm:
            nn.utils.spectral_norm(cls)
        self.cls = nn.Sequential(cls)

    def forward(self, x):
        z = self.feat_ext(x)
        y_hat = self.cls(z)

        return y_hat, z

