import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class SilhouetteLayer(nn.Module):

    def __init__(self, init_power=1., n_bins=100, lo=0., hi=1.):
        super().__init__()
        self.power = nn.parameter.Parameter(data=T.Tensor([init_power]))

        bins, _ = np.linspace(lo, hi, retstep=True, num=n_bins)
        self.bins = T.reshape(T.Tensor(bins), (1, 1, -1))
        self.lo = lo
        self.hi = hi

    def forward(self, x_dgms):

        channels = []
        for dim, dgms in x_dgms.items():

            births, deaths = dgms[:, :, [0]], dgms[:, :, [1]]
            weights = deaths - births

            weights = T.pow(weights, self.power)
            weights_sum = T.sum(weights, dim=1)

            weights_sum[weights_sum == 0.] = np.inf
            mid_points = (deaths + births) / 2
            heights = (deaths - births) / 2
            # print('Sub', self.bins.shape,
            #     mid_points.shape, (self.bins-mid_points).shape)
            fibers = F.relu(-T.abs(self.bins - mid_points) + heights)
            fibers_weighted_sum = T.sum(weights * fibers, axis=1) / weights_sum
            channels.append(fibers_weighted_sum[:, None, :])

        x_channels = T.cat(channels, dim=1)
        # x_channels.shape: (batch_size, dimension, sampled_values)
        return x_channels
