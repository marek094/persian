## 5-Layer CNN for CIFAR
## Based on https://myrtle.ai/learn/how-to-train-your-resnet-4-architecture/

import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


def make_cnn(c=64, num_classes=10):
    ''' Returns a 5-layer CNN with width parameter c. '''
    return nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(c, c * 2, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 2),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 4),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c * 8),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c * 8, num_classes, bias=True))


class CNN(nn.Module):
    def __init__(self, c=64, num_classes=10):
        super(CNN, self).__init__()

        self.conv0 = nn.Conv2d(3,
                               c,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.bn0 = nn.BatchNorm2d(c)
        # Layer 1
        self.conv1 = nn.Conv2d(c,
                               c * 2,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(c * 2)
        # Layer 2
        self.conv2 = nn.Conv2d(c * 2,
                               c * 4,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.bn2 = nn.BatchNorm2d(c * 4)
        # Layer 3
        self.conv3 = nn.Conv2d(c * 4,
                               c * 8,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=True)
        self.bn3 = nn.BatchNorm2d(c * 8)
        # Layer 4
        self.fc = nn.Linear(c * 8, num_classes, bias=True)

    def _features(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = nn.ReLU()(out)

        # Layer 1
        out = self.conv1(out)
        out = self.bn1(out)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(2)(out)

        # Layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(2)(out)

        # Layer 3
        out = self.conv3(out)
        out = self.bn3(out)
        out = nn.ReLU()(out)
        out = nn.MaxPool2d(2)(out)

        # Layer 4
        out = nn.MaxPool2d(4)(out)
        out = Flatten()(out)
        return out

    def forward(self, x):
        out = self._features(x)
        out = self.fc(out)

        return out
