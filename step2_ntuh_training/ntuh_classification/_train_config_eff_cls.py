

from train_eff_cls import *

pretrain_weight_path = '../../step1_pretraining/step2_classification/model_weights/eff_cls_siim/eff_cls_siim_'
import os
if not os.path.exists(pretrain_weight_path):
    raise ValueError(f"complete trained model weight not exist : {pretrain_weight_path}")

for i in range(5):
    train(max_epoch = 50, batch_size=8, num_samples=1000, fold_num=i, pt_path=pretrain_weight_path)