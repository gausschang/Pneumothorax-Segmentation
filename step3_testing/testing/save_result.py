import os
import cv2
import numpy as np
import monai

cmap_list = ['Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']
def save(img, pred, gt=None, cmap='cool', save_path='', save_name=''):
    
    for i in [0]:

        if not os.path.exists(save_path):
            os.makedirs(save_path)


        img1 = img[:,0,:,:].to('cpu').numpy()
        pred1 = pred.to('cpu').numpy().squeeze(axis=0)
        
        img1 = monai.visualize.utils.blend_images(img1, pred1, cmap=cmap, alpha=0.7, rescale_arrays=True)

        if gt is not None:
            gt = gt.cpu().numpy().squeeze(axis=0)
            img1 = monai.visualize.utils.blend_images(img1, gt, cmap='cool', alpha=0.5, rescale_arrays=True)

        img1 = np.moveaxis(img1, 0, -1)*255

        cv2.imwrite(os.path.join(save_path , f"{save_name}_{cmap}.png"), img1)

#   'RdPu'  'brg'   'cool'0.8  

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, RocCurveDisplay
from sklearn import metrics
import json

def auc(y_test, y_score, save_path, save_name, thres):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # conf matrx
    y_pred_binary = (y_score > thres)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)


    # AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    idx = np.argmax(tpr - fpr)

    with open(os.path.join(save_path, f'{save_name}.json'), "w") as f:
        json.dump({'auc':str(roc_auc), 'threshold': str(thresholds[idx]), 'fpr':str(fpr[idx]), 'tpr':str(tpr[idx]), "sensitivity":str(sensitivity), "specificity":str(specificity)}, f)
    plt.savefig(os.path.join(save_path, f'{save_name}.png'))
    plt.close()

def iou(pred, gt):
    i = (pred*gt).sum()
    u = ((pred+gt)>0).sum()
    iou = i/u
    print(iou)
    return iou