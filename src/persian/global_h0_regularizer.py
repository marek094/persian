import torch


class GlobalH0Regularizer():
    @staticmethod
    def nearest_query(data, keys):
        diff = data[None, :] - keys[:, None]
        dm = diff.norm(p=2, dim=2)
        return dm.min(axis=1).values

    @staticmethod
    def infs(*args):
        zeros = torch.zeros(*args, dtype=torch.float)
        return zeros + 1e8

    def __init__(self, data_size, feat_dim, nclasses, device, decay):
        # without gradients
        self.dev = device
        self.data = self.infs([data_size, feat_dim]).to(self.dev)
        self.labels = -torch.ones([data_size], dtype=torch.long).to(self.dev)
        self.nclasses = nclasses
        self.decay = decay

    def update(self, idcs, feat):
        self.data[idcs] = feat

    def criterium(self, idcs, labels, feats):
        self.update(idcs, self.infs(feats.shape).to(self.dev))
        self.labels[idcs] = labels
        result = [
            self.nearest_query(self.data[torch.where(self.labels == cl)],
                               feats[torch.where(labels == cl)])
            for cl in range(self.nclasses)
        ]
        self.update(idcs, feats.detach())
        return self.decay * torch.cat(result).mean()
