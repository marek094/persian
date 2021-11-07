from persian.main import experiment_builder, string_to_arggrid
from persian.trainer import saved_torch_training
from persian.errors import IncompatibleFlagsError

import os
import torch

torch.autograd.set_detect_anomaly(True)

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BEST_EXPERIMENT = """
    --schema prediction_pers
    --logdir rprs/3
    --no-ndalg
    --pers_type l1;l2
    --no-use_norm
    --lr 1e-5
    --lr_inp 1e-2
    --epochs 100
    --seed 2021
"""

if __name__ == "__main__":
    for args in string_to_arggrid(BEST_EXPERIMENT):
        try:
            model, _ = experiment_builder(default_args=args)
            saved_torch_training(model)
        except IncompatibleFlagsError as e:
            print('IncompatibleFlagsError: ', e)
