

from train_vit_cls import *

pretrain_weight_path = '../../step1_pretraining/step2_classification/model_weights/vit_cls_siim/vit_cls_siim_'
import os
if not os.path.exists(pretrain_weight_path):
    raise ValueError(f"complete trained model weight not exist : {pretrain_weight_path}")

for i in range(5):
    train(max_epoch = 50, batch_size=8, num_samples=1000, fold_num=i, pt_path=pretrain_weight_path)