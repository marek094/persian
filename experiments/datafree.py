from persian.main import experiment_builder, string_to_args
from persian.trainer import saved_torch_training
from persian.errors import IncompatibleFlagsError

import os
import torch

torch.autograd.set_detect_anomaly(True)

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

BEST_EXPERIMENT = """
    --schema prediction_pers
    --logdir rprs
    --no-ndalg
    --pers_type l2
    --no-use_norm
"""

if __name__ == "__main__":
    try:
        model, _ = experiment_builder(
            default_args=string_to_args(BEST_EXPERIMENT))
        saved_torch_training(model)
    except IncompatibleFlagsError as e:
        print('IncompatibleFlagsError:', e)
