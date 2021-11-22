from persian.main import experiment_builder, string_to_arggrid
from persian.trainer import saved_torch_training
from persian.errors.flags_incompatible import IncompatibleFlagsError

import os
import torch

torch.autograd.set_detect_anomaly(True)

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

EXPERIMENT_DEFAULT = """
    --logdir boundary
    --schema cnn_boundary

    --seed 2022
    --width 48

    --epochs 4000
    --batch_size 128
    --lr 0.0001
    --optim adam
    --noise 15
    --aug ddd

    --save g127
    --sched None
    --w_decay 0.0
    --h0_decay 0.0
"""

if __name__ == "__main__":
    for args in string_to_arggrid(EXPERIMENT_DEFAULT):
        try:
            model, _ = experiment_builder(default_args=args)
            saved_torch_training(model, ask=True)
        except IncompatibleFlagsError as e:
            print('IncompatibleFlagsError: ', e)
