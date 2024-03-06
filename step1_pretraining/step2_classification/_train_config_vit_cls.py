

from train_vit_cls import *

pretrain_weight_path = '../../step1_pretraining/step1_segmentation/model_weights/tinyvit_siim/tinyvit_siim_'
import os
if not os.path.exists(pretrain_weight_path):
    raise ValueError(f"complete trained model weight not exist : {pretrain_weight_path}")

train(max_epoch = 20, batch_size=8, num_samples=500, pt_path=pretrain_weight_path)