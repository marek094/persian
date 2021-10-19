from persian.main import experiment_builder, string_to_args
from persian.trainer import saved_torch_training

import os

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

sota_default = """
    --schema: cnn_simple
    --batch_size: 32
    --nlayers: 13
    --symetric
    
"""

if __name__ == "__main__":
    model, _ = experiment_builder(default_args=string_to_args(sota_default))
    saved_torch_training(model)

