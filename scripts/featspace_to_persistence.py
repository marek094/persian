import sys

from sklearn.externals._packaging.version import parse

sys.path.append('scripts')

from pathlib import Path
import numpy as np

from tqdm import tqdm
from gph import ripser_parallel

import minmax


def main(args):
    print('Hello world')

    run_folder = args.folder
    feat_paths = list(sorted(run_folder.glob('featspace_*.npz')))
    if args.test:
        feat_paths = feat_paths[:2]

    maxdim = 1
    dgms = []
    shape = []
    for feat_path in tqdm(feat_paths):
        x = np.load(feat_path)
        feats, labels = x['feats'], x['labels']
        dist_mat = minmax.get_matrix(labels, feats)
        dgm = ripser_parallel(
            dist_mat,
            maxdim=maxdim,
            n_threads=6,
        )
        loc_dgms = dgm['dgms']
        dgms += list(loc_dgms)

        loc_shape = [len(d) for d in loc_dgms] + [0] * (maxdim + 1)
        shape.append(loc_shape[:maxdim + 1])

    dgms_a = np.concatenate(dgms)
    shape_a = np.array(shape).astype(np.uint64)
    np.savez(run_folder / 'dgms.npz', dgms=dgms_a, shape=shape_a)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', type=Path)
    parser.add_argument('-j', type=int, default=-1)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    assert args.folder is not None
    assert args.folder.exists()

    main(args)
