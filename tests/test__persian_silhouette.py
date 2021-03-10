import unittest

import numpy as np
from gtda.diagrams import Silhouette as GiottoSilhouette
from torch import Tensor

from persian.layer_silhouette import SilhouetteLayer
from persian.schema_tda import TdaSchema


class TestSilhouette(unittest.TestCase):

    @staticmethod
    def gen_dgm_batch(n_pts=200, batch_size=8, dims=3):

        def gen_dgm_dim(n_pts):
            db = np.random.random_sample((batch_size, n_pts, 2))
            # include min/max values for compatibility with gtda
            db[:, -1, 0], db[:, -1, 1] = 0., 1.
            db = db / 5
            return np.sort(db, axis=2)

        return {dim: gen_dgm_dim(n_pts=n_pts // 2**dim) for dim in range(dims)}

    @staticmethod
    def batch_to_gtda(dgms):
        n_batches = len(dgms[0])
        result = []
        for i_batch in range(n_batches):
            batch = []
            for dim, d in dgms.items():
                n_pts, two = d[i_batch].shape
                assert two == 2

                dd = np.ones((n_pts, 3)) * dim
                dd[:, :2] = d[i_batch]
                batch.append(dd)
            result.append(np.concatenate(batch, axis=0))

        return np.array(result)

    def test_torch_layer(self):
        dgms = TestSilhouette.gen_dgm_batch()

        gs = GiottoSilhouette(
            power=0.01,
            n_bins=64,
        )

        g_result = gs.fit_transform(TestSilhouette.batch_to_gtda(dgms))

        sl = SilhouetteLayer(init_power=0.01, n_bins=64, lo=0, hi=0.2)
        sl_result = sl({d: Tensor(t) for d, t in dgms.items()})
        sl_result = sl_result.detach().numpy()

        self.assertTrue(np.allclose(sl_result, g_result))

    def test_transformations(self):

        dgm = np.array([
            (1, 2, 0),
            (3, 4, 0),
            (5, 6, 0),
            (2, 3, 1),
            (2, 5, 1),
            (2, 3, 2),
            (2, 3, 2),
            (3, 5, 2),
        ],
                       dtype=float)

        res = TdaSchema.dgm_from_gtda(dgm)
        self.assertTrue(np.allclose(res[0], dgm[0:3, :2]))
        self.assertTrue(np.allclose(res[1], dgm[3:5, :2]))
        self.assertTrue(np.allclose(res[2], dgm[5:8, :2]))


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
