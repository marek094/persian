from persian.schemas.torch_zooset import ZoosetTorchSchema
from persian.errors import UnknownFlagValueError

import torch as T
import numpy as np
from torchph.pershom import vr_persistence_l1, vr_persistence


class PersPredictionSchema(ZoosetTorchSchema):
    @staticmethod
    def list_hparams():
        return ZoosetTorchSchema.list_hparams() + [
            dict(name='epochs', type=int, default=200),
            dict(name='max_dim', type=int, default=1),
            dict(name='npts', type=int, default=64),
            dict(name='lr_inp', type=float, default=1e-2),
            dict(name='lr', type=float, default=1e-5),
            dict(name='dgm_limit', type=int, default=2000),
            dict(name='dropout', type=float, default=0.15),
            dict(name='pers_type', type=str, default='l1'),
            dict(name='use_norm', type=bool, default=True),
            dict(name='gamma', type=float, default=0.9),
            dict(name='width', type=int, default=1024),
            dict(name='dim_inp', type=int, default=32),  # bad name
            dict(name='padd_inp', type=float, default=0.),
            dict(name='init_inp', type=str, default='u'),
            dict(name='gamma_step', type=int, default=1),
            dict(name='batch_norm', type=int, default=0),
            dict(name='weight_decay', type=float, default=0),
            dict(name='graph_mode', type=bool, default=False),
            dict(name='padding_input', type=int, default=0),
        ]

    def __init__(self, flags={}):
        super().__init__(flags)

        if flags['dim_inp'] + 2 * flags['padding_input'] not in [
                16, 28, 32, 36
        ]:
            raise UnknownFlagValueError('Input dim value is not allowed')

    def prepare_model(self):
        npts = self.flags['npts']
        dim = self.flags['dim_inp']
        ppnet = PersistenceForPredictionNet(
            npts=npts,
            max_dim=self.flags['max_dim'],
            dev=self.dev,
            limit=self.flags['dgm_limit'],
            pers=self.flags['pers_type'],
            use_norm=self.flags['use_norm'],
            input_space_shape=[npts, 1, dim, dim],
            input_values_padding=self.flags['padd_inp'],
            initialization=self.flags['init_inp'],
            learnable=not self.flags['graph_mode'],
            flags=self.flags,
        )
        if self.flags['use_norm']:
            iface = 4190
        else:
            iface = 2 * self.flags['dgm_limit'] + 2 * (npts - 1)

        def get_bn(feats):
            if self.flags['batch_norm'] == 1:
                return [T.nn.BatchNorm1d(num_features=feats, device=self.dev)]
            return []

        use_bias = self.flags['batch_norm'] == 0
        width = self.flags['width']

        # yapf: disable
        model = T.nn.Sequential(*[
            ppnet,
            T.nn.Flatten(),
            ] + get_bn(iface) + [

            T.nn.Linear(iface, 4 * width, bias=use_bias),
            ] + get_bn(4 * width) + [
            T.nn.LeakyReLU(),

            T.nn.Linear(4 * width, 2 * width, bias=use_bias),
            ] + get_bn(2 * width) + [
            T.nn.LeakyReLU(),

            T.nn.Linear(2 * width, 4 * width, bias=use_bias),
            ] + get_bn(4 * width) + [
            T.nn.LeakyReLU(),

            T.nn.Linear(4 * width, width, bias=use_bias),
            ] + get_bn(width) + [
            T.nn.LeakyReLU(),

            T.nn.Dropout(self.flags['dropout'], inplace=True),
            T.nn.Linear(width, 1),
            T.nn.Sigmoid()
        ])


        # yapf: enable
        self.model = model.to(self.dev)

    def prepare_criterium(self):
        modules = list(self.model.modules())
        self.optim = T.optim.Adam([{
            'params': module.parameters()
        } for module in modules[2:]] + [{
            'params': modules[1].parameters(),
            "lr": self.flags['lr_inp']
        }],
                                  lr=self.flags['lr'],
                                  weight_decay=self.flags['weight_decay'])

        self.scheduler = T.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=self.flags['gamma_step'],
            gamma=self.flags['gamma'],
        )

        self.crit = T.nn.MSELoss().to(self.dev)

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        func_batches = {
            'TRAIN': self._run_batches_train,
            'VALID': self._run_batches_valid,
        }

        assert set_name in func_batches
        self.metrics[set_name] = func_batches[set_name](set_name)

    def run_inference(self, input):
        raise NotImplementedError()

    def _run_batches_train(self, set_name):
        assert set_name in self.loaders
        self.model.train()
        running_loss = 0
        all_targets = []
        total = 0

        for inputs, targets_ in self.loaders[set_name]:
            targets = targets_.to(self.dev)

            self.optim.zero_grad()
            outputs = self.model(inputs)

            loss = self.crit(outputs, targets)
            loss.backward()
            self.optim.step()

            running_loss += loss.item()
            all_targets += list(targets_[:, 0])
            total += targets.size(0) / self.flags['batch_size']

        self.scheduler.step()
        # print(all_targets)
        loss = running_loss / total
        r2 = 1.0 - loss / np.var(all_targets)

        return dict(loss=loss, r2=r2)

    def _run_batches_valid(self, set_name):
        self.model.eval()
        running_loss = 0
        all_targets = []
        total = 0

        with T.no_grad():
            for inputs, targets_ in self.loaders[set_name]:
                targets = targets_.to(self.dev)
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)
                running_loss += loss.item()
                all_targets += list(targets_[:, 0])
                total += targets.size(0) / self.flags['batch_size']

        # print(all_targets)
        loss = running_loss / total
        r2 = 1.0 - loss / np.var(all_targets)

        return dict(loss=loss, r2=r2)


class PersistenceForPredictionNet(T.nn.Module):
    @staticmethod
    def _gen_input(shape, input_values_padding):
        shape0 = (shape[0], np.product(shape[1:]))
        u = np.random.normal(0 - input_values_padding,
                             1 + input_values_padding, shape0)
        norm = np.linalg.norm(u, axis=1, keepdims=True)
        return (u / norm).reshape(shape).astype(np.float32)

    def __init__(self,
                 npts,
                 max_dim,
                 dev,
                 limit,
                 use_norm=True,
                 pers='l1',
                 input_space_shape=[128, 1, 32, 32],
                 concat_input_space=True,
                 input_values_padding=0,
                 initialization='u',
                 learnable=True,
                 flags=None):
        super().__init__()

        self.npts = npts
        self.max_dim = max_dim
        self.dev = dev
        self.limit = limit
        self.use_norm = use_norm
        self.learnable = learnable

        assert pers in ['l1', 'l2']
        self.pers = pers

        if initialization == 'u':
            tsr = T.tensor(
                self._gen_input(input_space_shape, input_values_padding))
        elif initialization.startswith('o'):
            tsr = T.empty(input_space_shape)
            T.nn.init.orthogonal_(tsr, gain=float(initialization[1:]))
        elif initialization.startswith('k'):
            tsr = T.empty(input_space_shape)
            mode = dict(i='fan_in', o='fan_out')[initialization[1]]
            T.nn.init.kaiming_uniform_(tsr, mode=mode, nonlinearity='relu')
        else:
            assert False

        if flags['padding_input'] > 0:
            p = flags['padding_input']
            tsr = T.nn.functional.pad(tsr, (p, p, p, p), 'constant', 0)

        self.input_space = T.nn.Parameter(tsr, requires_grad=learnable)

        self.batch = input_space_shape[0]
        self.concat_input_space = concat_input_space

    @staticmethod
    def _flatcat(inps):
        bs = inps[0].shape[0]
        return T.cat([T.reshape(a, (bs, -1)) for a in inps], axis=1)

    @staticmethod
    def _flatcat2(inps):
        return T.cat([T.reshape(a, (-1, )) for a in inps], axis=0)

    @staticmethod
    def pers_l2(feats, max_dimension):
        n = feats.size(0)
        x = feats.unsqueeze(1).expand(
            [feats.size(0), feats.size(0),
             feats.size(1)])
        x = x.transpose(0, 1) - x
        x = x.norm(dim=2, p=2)
        return vr_persistence(x, max_dimension=max_dimension)

    def _pershom(self, space):
        if self.pers == 'l1':
            ph = vr_persistence_l1(
                space,
                max_dimension=self.max_dim,
            )
        elif self.pers == 'l2':
            ph = self.pers_l2(
                space,
                max_dimension=self.max_dim,
            )
        return ph[0]

    def _vect(self, pers_hom):
        assert len(pers_hom) == self.max_dim + 1

        def padd(v, n: int):
            zs = T.zeros((n, 2)).to(self.dev)
            nv = min(len(v), n)
            zs[:nv] = v[:nv]
            return zs

        return T.cat([
            padd(v, n) for v, n in zip(pers_hom, [
                self.npts - 1,
                self.limit,
                self.limit * 2,
            ])
        ],
                     axis=0)

    def forward(self, x_cnn_batch):
        if self.use_norm:
            pers_hom = [
                self._flatcat2([
                    T.norm(self._flatcat([
                        self.input_space,
                    ]),
                           dim=1,
                           keepdim=True),
                    self._vect(
                        self._pershom(
                            self._flatcat([
                                x_cnn(self.input_space),
                            ]), ), ),
                ]) for x_cnn in x_cnn_batch
            ]
        # elif self.learnable:
        #     pers_hom = [
        #         self._flatcat2([
        #             self._flatcat([
        #                 self.input_space,
        #             ]),
        #             self._vect(
        #                 self._pershom(
        #                     self._flatcat([
        #                         x_cnn(self.input_space),
        #                     ]), ), ),
        #         ]) for x_cnn in x_cnn_batch
        #     ]
        else:
            pers_hom = [
                self._vect(
                    self._pershom(self._flatcat([
                        x_cnn(self.input_space),
                    ]), ), ) for x_cnn in x_cnn_batch
            ]
        return T.stack(pers_hom, axis=0)
