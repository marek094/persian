from persian.main import experiment_builder, string_to_arggrid
from persian.trainer import validated_training
from persian.errors.flags_incompatible import IncompatibleFlagsError

import os
import torch

torch.autograd.set_detect_anomaly(True)

# deterministic behauvior for NVIDIA 30*
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

SOTA_DEFAULT = """
    --logdir runs/v3
    --schema cnn_simple
    --train_size 500
    --batch_size 256
    --optim sgd
    --nlayers 13
    --symetric
    --dropout 0.5
    --lr 0.5;0.3
    --w_decay 1e-3
    --ndalg
    --sub_batches 16
    --h0_decay 0.1
    --h0_dens 0.7
    --aug tdd
    --sched cos
    --epochs 310
"""

# SimpleCnnSchema_lruns,s2022,nTrue,b256,o0,dcifar10,t500,u16,atdd,e310,r0.1,psgd,ccos,w0.001,y0.1,A13,i128,mTrue,B0.5_

if __name__ == "__main__":
    for args in string_to_arggrid(SOTA_DEFAULT):
        try:
            model, _ = experiment_builder(default_args=args)
            validated_training(model, save_txt=True)
        except IncompatibleFlagsError as e:
            print('IncompatibleFlagsError: ', e)
