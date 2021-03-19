from persian.trainer import validated_training
from persian.schema_tda import TdaSchema
from persian.torchize import SklearnTranform
from persian.schema_expt2 import Flatten

from pathlib import Path
import gtda.diagrams as gvect

import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import torch as T
import torch.nn as nn
import torch.nn.functional as F


class SmallZooPaper:
    DATAFRAME_CONFIG_COLS = [
        'config.w_init', 'config.activation', 'config.learning_rate',
        'config.init_std', 'config.l2reg', 'config.train_fraction',
        'config.dropout'
    ]
    CATEGORICAL_CONFIG_PARAMS = ['config.w_init', 'config.activation']
    CATEGORICAL_CONFIG_PARAMS_PREFIX = ['winit', 'act']
    DATAFRAME_METRIC_COLS = [
        'test_accuracy', 'test_loss', 'train_accuracy', 'train_loss'
    ]
    TRAIN_SIZE = 15000

    @staticmethod
    def filter_checkpoints(dataframe,
                           target='test_accuracy',
                           stage='final',
                           binarize=True):
        """Take one checkpoint per run and do some pre-processing.

        Args:
            weights: numpy array of shape (num_runs, num_weights)
            dataframe: pandas DataFrame which has num_runs rows. First 4 columns should
            contain test_accuracy, test_loss, train_accuracy, train_loss respectively.
            target: string, what to use as an output
            stage: flag defining which checkpoint out of potentially many we will take
            for the run.
            binarize: Do we want to binarize the categorical hyperparams?

        Returns:
            tuple (weights_new, metrics, hyperparams, ckpts), where
            weights_new is a numpy array of shape (num_remaining_ckpts, num_weights),
            metrics is a numpy array of shape (num_remaining_ckpts, num_metrics) with
                num_metric being the length of DATAFRAME_METRIC_COLS,
            hyperparams is a pandas DataFrame of num_remaining_ckpts rows and columns
                listed in DATAFRAME_CONFIG_COLS.
            ckpts is an instance of pandas Index, keeping filenames of the checkpoints
            All the num_remaining_ckpts rows correspond to one checkpoint out of each
            run we had.
        """
        COLS = SmallZooPaper.DATAFRAME_CONFIG_COLS + SmallZooPaper.DATAFRAME_METRIC_COLS
        for t in target:
            assert t in COLS, 'unknown target'

        ids_to_take = []
        # Keep in mind that the rows of the DataFrame were sorted according to ckpt
        # Fetch the unit id corresponding to the ckpt of the first row
        current_uid = dataframe.axes[0][0].split('/')[-2]  # get the unit id
        steps = []
        for i in range(len(dataframe.axes[0])):
            # Fetch the new unit id
            ckpt = dataframe.axes[0][i]
            parts = ckpt.split('/')
            if parts[-2] == current_uid:
                steps.append(int(parts[-1].split('-')[-1]))
            else:
                # We need to process the previous unit
                # and choose which ckpt to take
                steps_sort = sorted(steps)
                target_step = -1
                if stage == 'final':
                    target_step = steps_sort[-1]
                elif stage == 'early':
                    target_step = steps_sort[0]
                else:  # middle
                    target_step = steps_sort[int(len(steps) / 2)]
                offset = [
                    j for (j, el) in enumerate(steps) if el == target_step
                ][0]
                # Take the DataFrame row with the corresponding row id
                ids_to_take.append(i - len(steps) + offset)
                current_uid = parts[-2]
                steps = [int(parts[-1].split('-')[-1])]

        # Fetch the hyperparameters of the corresponding checkpoints
        hyperparams = dataframe[SmallZooPaper.DATAFRAME_CONFIG_COLS]
        hyperparams = hyperparams.iloc[ids_to_take]
        if binarize:
            # Binarize categorical features
            hyperparams = pd.get_dummies(
                hyperparams,
                columns=SmallZooPaper.CATEGORICAL_CONFIG_PARAMS,
                prefix=SmallZooPaper.CATEGORICAL_CONFIG_PARAMS_PREFIX)
        else:
            # Make the categorical features have pandas type "category"
            # Then LGBM can use those as categorical
            hyperparams.is_copy = False
            for col in SmallZooPaper.CATEGORICAL_CONFIG_PARAMS:
                hyperparams[col] = hyperparams[col].astype('category')

        # Fetch the file paths of the corresponding checkpoints
        ckpts = dataframe.axes[0][ids_to_take]

        return (ids_to_take,
                dataframe[target].values[ids_to_take, :].astype(np.float32),
                hyperparams, ckpts)

    @staticmethod
    def split(metrics_file: Path, train_ratio: float, target='test_accuracy'):
        # TODO(mcerny): add ratio param
        assert metrics_file.exists()
        dgms_folder = metrics_file.parent

        metrics = pd.read_csv(metrics_file, index_col=0)
        target = target if isinstance(target, list) else [target]

        ids_to_take, metrics_flt, configs_flt, ckpts = SmallZooPaper.filter_checkpoints(
            metrics,
            target=target,
            binarize=True,
            stage='final',
        )

        train_size = round(len(ids_to_take) * train_ratio)

        # Filter out DNNs with NaNs and Inf in the weights
        # idx_valid = (np.isfinite(weights_flt).mean(1) == 1.0)
        # inputs = np.asarray(weights_flt[idx_valid], dtype=np.float32)
        # outputs = np.asarray(metrics_flt[idx_valid], dtype=np.float32)
        # configs = configs_flt.iloc[idx_valid]
        # ckpts = ckpts[idx_valid]
        outputs = metrics_flt
        configs = configs_flt

        # Shuffle and split the data
        random_idx = list(range(outputs.shape[0]))
        np.random.shuffle(random_idx)
        # weights_train[dataset], weights_test[dataset] = (
        #     inputs[random_idx[:TRAIN_SIZE]], inputs[random_idx[TRAIN_SIZE:]])
        outputs_train, outputs_valid = (1. * outputs[random_idx[:train_size]],
                                        1. * outputs[random_idx[train_size:]])

        configs_train, configs_valid = (configs.iloc[random_idx[:train_size]],
                                        configs.iloc[random_idx[train_size:]])

        dgms_files = [
            dgms_folder / f'dgm_i{id_:06},s94,c1_.npz' for id_ in ids_to_take
        ]
        dgms_train, dgms_valid = ([
            dgms_files[idx] for idx in random_idx[:train_size]
        ], [dgms_files[idx] for idx in random_idx[train_size:]])

        print(outputs_train.shape)

        return dgms_train, outputs_train, dgms_valid, outputs_valid

    def build_fcn(n_layers,
                  n_hidden,
                  n_outputs,
                  dropout_rate,
                  activation,
                  w_regularizer,
                  w_init,
                  b_init,
                  last_activation='softmax'):
        """Fully connected deep neural network."""
        model = []
        # model.append(Flatten())
        # for _ in range(n_layers):
        #     model.add(
        #         nn.Linear(
        #             n_hidden,
        #             activation=activation,
        #             kernel_regularizer=w_regularizer,
        #             kernel_initializer=w_init,
        #             bias_initializer=b_init))
        #     if dropout_rate > 0.0:
        #     model.add(keras.layers.Dropout(dropout_rate))
        # if n_layers > 0:
        #     model.add(keras.layers.Dense(n_outputs, activation=last_activation))
        # else:
        #     model.add(keras.layers.Dense(
        #         n_outputs,
        #         activation='sigmoid',
        #         kernel_regularizer=w_regularizer,
        #         kernel_initializer=w_init,
        #         bias_initializer=b_init))
        # return model


class BenchmarkSchema(TdaSchema):

    @staticmethod
    def list_hparams():
        return TdaSchema.list_hparams() + [
            dict(name='datadir', type=Path, default='../zoo_dgms'),
            dict(name='split_seed', type=int, default=42),
            dict(name='split_ratio', type=float, default=0.5),
            dict(name='split_func', type=str, default='gen'),
            dict(name='target', type=str, default='test_accuracy'),
            dict(name='vect', type=str, default='silhM'),
            dict(name='bins', type=int, default=64),
            dict(name='epochs', type=int, default=30),
            dict(name='batch_size', type=int, default=8),
            dict(name='lr', type=float, default=0.01),
        ]

    @staticmethod
    def list_zoo_params(df):
        dfd = df[df['step'] == 0]
        config = {}
        for col in dfd.columns:
            cs = col.split('.')
            if cs[0] == 'config':
                config[cs[1]] = dfd[col].unique()
        return config

    @staticmethod
    def _vectorization_by_name(name: str, n_bins=None):
        if name == 'landscape':
            return dict(dim=1,
                        tr=SklearnTranform(
                            gvect.PersistenceLandscape,
                            n_bins=n_bins,
                        ))

        if name == 'betti':
            return dict(dim=1,
                        tr=SklearnTranform(
                            gvect.BettiCurve,
                            n_bins=n_bins,
                        ))

        if name.startswith('heat'):
            return dict(dim=2,
                        tr=SklearnTranform(gvect.HeatKernel,
                                           n_bins=n_bins,
                                           sigma=dict(S=0.02, M=0.1,
                                                      L=0.8)[name[-1]]))

        if name.startswith('image'):
            return dict(dim=2,
                        tr=SklearnTranform(gvect.PersistenceImage,
                                           n_bins=n_bins,
                                           sigma=dict(S=0.02, M=0.1,
                                                      L=0.8)[name[-1]]))

        if name.startswith('silh'):
            return dict(dim=1,
                        tr=SklearnTranform(gvect.Silhouette,
                                           n_bins=n_bins,
                                           power=dict(S=0.003, M=0.1,
                                                      L=3)[name[-1]]))

        if name == 'template':
            # loading pervect is very slow
            import pervect as pvect
            return dict(dim=1, tr=SklearnTranform(pvect.PersistenceVectorizer))

        raise KeyError(name)

    def __init__(self, flags) -> None:
        super().__init__(flags=flags)

        self.vect = self._vectorization_by_name(self.flags['vect'],
                                                self.flags['bins'])

    def prepare_dataset(self, set_name):
        sets = ['TRAIN', 'VALID']
        assert set_name in sets
        if len(self.loaders) == 2:
            return

        metrics_file = self.flags['datadir'] / 'metrics.csv'
        np.random.seed(self.flags['seed'])
        train_X, train_y, valid_X, valid_y = SmallZooPaper.split(
            metrics_file=metrics_file,
            train_ratio=self.flags['split_ratio'],
        )

        self.dataset_var = {}

        for set_name, (X, y) in zip(sets, [(train_X, train_y),
                                           (valid_X, valid_y)]):
            ds = TdaSchema.DgmDataset(data=zip(X, y),
                                      transform=self.vect['tr'],
                                      load_all=True,
                                      from_gtda=False)

            self.dataset_var[set_name] = np.mean((y - np.mean(y))**2)
            bs = self.flags['batch_size']
            is_train = set_name == 'TRAIN'
            self.loaders[set_name] = DataLoader(ds,
                                                batch_size=bs,
                                                shuffle=is_train)

    def prepare_model(self):
        W = self.flags['bins']
        if self.vect['dim'] == 1:
            model = nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=9, stride=4),\
                nn.BatchNorm1d(8),\
                nn.ReLU(),\
                nn.Conv1d(8, 16, kernel_size=5, stride=2),\
                nn.BatchNorm1d(16),\
                nn.ReLU(),\
                nn.Conv1d(16, 32, kernel_size=5, stride=2),\
                nn.BatchNorm1d(32),\
                nn.ReLU(),\
                nn.Flatten(),\
                nn.Linear(W // 2, 1),\
                nn.Sigmoid())
        else:
            ...

        # print(np.sum([np.prod(val.shape)
        # for param, val in model.state_dict().items()]))
        # exit(0)
        self.model = model.to(self.dev)

    def prepare_criterium(self):
        self.optim = T.optim.SGD(
            self.model.parameters(),
            lr=self.flags['lr'],
            momentum=0.9,
            weight_decay=0.,
        )

        self.sched = T.optim.lr_scheduler.StepLR(
            self.optim,
            step_size=1,
            # gamma=self.flags['gamma'],
        )

        self.crit = nn.MSELoss().to(self.dev)

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        self.metrics[set_name] = dict(
            TRAIN=self._run_batches_train,
            VALID=self._run_batches_valid,
        )[set_name](set_name)

    def _run_batches_train(self, set_name) -> None:
        self.model.train()
        train_loss, train_mae, total = 0, 0, 0
        for inputs, targets in self._placed_loader(set_name):
            self.optim.zero_grad()
            outputs = self.model(inputs)
            # print("%@", outputs.shape, targets.shape)
            # print("%@S", outputs, targets)
            loss = self.crit(outputs, targets)
            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            train_mae += T.mean(T.abs(outputs - targets))
            total += targets.size(0)

            self.sched.step()

        r1inv = train_loss / self.dataset_var[set_name]
        return dict(
            mse=train_loss / total,
            mae=train_mae / total,
            r1=1.0 - r1inv / total,
        )

    def _run_batches_valid(self, set_name):
        self.model.eval()
        valid_loss, valid_mae, total = 0, 0, 0
        with T.no_grad():
            for inputs, targets in self._placed_loader(set_name):
                outputs = self.model(inputs)
                loss = self.crit(outputs, targets)

                valid_loss += loss.item()
                valid_mae += T.mean(T.abs(outputs - targets))
                total += targets.size(0)

        r1inv = valid_loss / self.dataset_var[set_name]
        return dict(
            mse=valid_loss / total,
            mae=valid_mae / total,
            r1=1.0 - r1inv / total,
        )


if __name__ == "__main__":
    # params = BenchmarkSchema.list_zoo_params(pd.read_csv(Path('..') / 'zoo_dgms' / 'metrics.csv'))
    # for p, v in params.items():
    #     print(f'{p}'.ljust(40), f'{len(v)}'.rjust(8), v)
    args = BenchmarkSchema.build_parser().parse_args()
    schema = BenchmarkSchema.from_args(args)
    validated_training(schema, verbose=True, save_txt=True)
