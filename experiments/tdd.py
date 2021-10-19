from persian.main import experiment_builder
from persian.trainer import saved_torch_training

import os

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

if __name__ == "__main__":
    model, _ = experiment_builder()
    saved_torch_training(model)

