from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from gtda.homology import VietorisRipsPersistence

from numpy.lib.index_tricks import AxisConcatenator


def layerwise_cloud(path: Path,
                    n_samples=None,
                    seed=None,
                    layer_norm=False,
                    count=1):
    assert path.exists()

    if seed is not None:
        np.random.seed(seed)

    if path.suffix == '.pt':
        import torch as T
        model_params = T.load(path, map_location=T.device('cpu'))
    else:
        raise NotImplementedError(f'Unsupported suffix {path.suffix}.')

    if not layer_norm:
        norm = lambda x: x
    else:
        norm = lambda x: (x - x.mean(axis=0)) / x.std(axis=0)

    conv_fc_only = [
        norm(v.numpy().reshape(-1))
        for v in model_params.values()
        if len(v.shape) > 1
    ]

    if n_samples is None:
        n_samples = min(v.shape[0] for v in conv_fc_only)

    return [
        np.array([np.random.choice(w, n_samples)
                  for w in conv_fc_only]).T
        for _ in range(count)
    ]


def persistet_diagram(clouds, n_jobs=1, dimensions=(0, 1, 2)):
    vrp = VietorisRipsPersistence(
        metric='euclidean',
        metric_params={},
        homology_dimensions=dimensions,
        n_jobs=n_jobs,
    )

    dgms = vrp.fit_transform(clouds)
    return dgms


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--files', nargs='*', type=Path, default=[])
    parser.add_argument('--jobs', '-j', type=int, default=8)
    args = parser.parse_args()

    for file_path in args.files:
        clouds = layerwise_cloud(file_path, n_samples=256, count=4)
        r = persistet_diagram(clouds, n_jobs=args.jobs)
