from persian.main import experiment_builder, string_to_args
from persian.trainer import saved_torch_training

import os

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sota_default = """
    --schema cnn_simple
    --train_size 500
    --batch_size 32
    --optim sgd
    --nlayers 13
    --symetric
    --dropout 0.5
    --lr 0.5
    --w_decay 1e-3
    --no-ndalg
    --sub_batches 8
    --h0_decay 0.1
"""

if __name__ == "__main__":
    model, _ = experiment_builder(default_args=string_to_args(sota_default))
    saved_torch_training(model)
