from persian.main import experiment_builder, string_to_arggrid
from persian.trainer import saved_torch_training

import os
import torch

torch.autograd.set_detect_anomaly(True)

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

SOTA_DEFAULT = """
    --schema cnn_simple
    --train_size 500;1000;4000
    --batch_size 8;128
    --optim sgd
    --nlayers 13
    --symetric
    --dropout 0.5
    --lr 0.5;0.3
    --w_decay 1e-3
    --ndalg
    --sub_batches 8;16
    --h0_decay 0.1
    --aug tdd
    --sched cos
"""

if __name__ == "__main__":
    for args in string_to_arggrid(SOTA_DEFAULT):
        model, _ = experiment_builder(default_args=args)
        saved_torch_training(model)
