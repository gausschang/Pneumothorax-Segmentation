import tqdm
import numpy as np
import torch
from get_inferer import *
from save_result import *
#import matplotlib.pyplot as plt

def predict(test_loader, cls_config, seg_config, save_path, save_info):
    inferer_cls = get_inferrer_cls(cls_config)
    inferer_seg = get_inferrer_seg(seg_config)
    y_pred = []
    y = []
    detectable = []
    for data in tqdm.tqdm(test_loader):
        
        if data["Source"][0] == 'ntuh' or data["Source"][0] == 'ntuh_e':
            images = data["image"].to('cuda', dtype=torch.float)
            label = data['label']

            _, aux = inferer_cls(images)

            y.append(label.numpy().astype('int'))
            y_pred.append(aux.numpy())

            if data['label'] == 1:
                gt_mask = data['mask'].to('cpu')
                pred_mask = (inferer_seg(images).to('cpu'))>0.5
                detect_size = (pred_mask*gt_mask).sum()
                if detect_size>0:
                    detectable.append(detect_size)

    
    y = np.hstack(y).squeeze(0)
    y_pred = np.hstack(y_pred).squeeze(0)
    detectable = np.array(detectable).min()
    #auc(y, y_pred, save_path, save_info)


    return y, y_pred, detectable