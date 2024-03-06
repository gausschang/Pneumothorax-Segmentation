

from train_eff_seg import *

pretrain_weight_path = '../../step1_pretraining/step1_segmentation/model_weights/efficientnetv2s_siim/efficientnetv2s_siim_'
import os
if not os.path.exists(pretrain_weight_path):
    raise ValueError(f"complete trained model weight not exist : {pretrain_weight_path}")

for i in range(5):
    train(max_epoch=100, fold_num=i, pt_path=pretrain_weight_path)