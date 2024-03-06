import os
import cv2
import numpy as np
import monai

def save(img, pred, gt=None, save_path='', save_name=''):

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    img = img[:,0,:,:].to('cpu').numpy()
    pred = pred.to('cpu').numpy().squeeze(axis=0)
    
    img = monai.visualize.utils.blend_images(img, pred, cmap='cool', alpha=0.8, rescale_arrays=True)

    if gt is not None:
        gt = gt.cpu().numpy().squeeze(axis=0)
        img = monai.visualize.utils.blend_images(img, gt, cmap='cool', alpha=0.5, rescale_arrays=True)

    img = np.moveaxis(img, 0, -1)*255

    cv2.imwrite(os.path.join(save_path , f"{save_name}.png"), img)

#   'RdPu'  'brg'   'cool'0.8  

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import json

def auc(y_test, y_score, seg_threshold, save_dir, save_name):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    idx = np.argmax(tpr - fpr)

    with open(os.path.join(save_dir, f'{save_name}.json'), "w") as f:
        json.dump({'auc':str(roc_auc), 'cls_threshold': str(thresholds[idx]), 'fpr':str(fpr[idx]), 'tpr':str(tpr[idx]), 'seg_threshold':str(seg_threshold)}, f)
    plt.savefig(os.path.join(save_dir, f'{save_name}.png'))
    plt.close()