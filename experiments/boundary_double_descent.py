from persian.main import experiment_builder, string_to_arggrid
from persian.trainer import validated_training
from persian.errors.flags_incompatible import IncompatibleFlagsError

import os
import torch

torch.autograd.set_detect_anomaly(True)

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

EXPERIMENT_DEFAULT = """
    --logdir boundary
    --schema cnn_boundary

    --epochs 4000
    --train_size 500
    --batch_size 256
    --lr 0.5
    --w_decay 1e-3
    --sched cos
    --epochs 310


    --optim adam
    --noise 15
    --aug ddd

"""

if __name__ == "__main__":
    for args in string_to_arggrid(EXPERIMENT_DEFAULT):
        try:
            model, _ = experiment_builder(default_args=args)
            validated_training(model, save_txt=True)
        except IncompatibleFlagsError as e:
            print('IncompatibleFlagsError: ', e)
