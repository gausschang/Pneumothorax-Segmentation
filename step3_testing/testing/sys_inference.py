import tqdm
import pickle
import numpy as np
import torch
from get_inferer import *
from save_result import *
import shutil

def di(mask, gt):
    d = monai.metrics.compute_meandice(mask>0.5, gt).item()
    d = round(d, 5)
    return d

def predict(test_loader, cls_config, seg_config, youden, seg_threshold, cmap, save_path, describ):

    # remove previous inference picture
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    inferer_cls = get_inferrer_cls(cls_config)
    inferer_seg = get_inferrer_seg(seg_config)
    y_pred = []
    y = []
    dice_list = {'dice':[], 'size':[]}
    tp_dice = []
    tp_dice_size = []
    tp_id = []
    cls_size = []
    fp_list = []
    for data in tqdm.tqdm(test_loader):

        images = data["image"].to('cuda', dtype=torch.float)
        gt = data["mask"]
        label = data['label']
        lesion_size = gt.sum()/(gt.shape[-1]*gt.shape[-2])

        dummy, prob = inferer_cls(images)
        predict_pos = (prob>youden)
        id = data['id'][0]

        
        # for complete segmentation ana (all positive cases)
        if (data["label"][0] == 1):
            mask = inferer_seg(images)
            d = di(mask, gt)
            dice_list['dice'].append(d)
            dice_list['size'].append(lesion_size)
        

        # tp (by classification model)
        if (data["label"][0] == 1) and (predict_pos==True):
            # segmentation
            mask = inferer_seg(images)
            d = di(mask, gt)
            
            # fn | classify true but seg false
            if (mask>0.5).sum()<seg_threshold:
                prob = prob*0
                save(images, dummy*0, gt=None, cmap=cmap, save_path=save_path +'/fn', save_name=f'{id}')
            
            # tp
            else:
                dice_score = monai.metrics.compute_meandice(mask>0.5, gt).item()
                tp_dice.append(dice_score)
                tp_dice_size.append(lesion_size)
                tp_id.append(id)
                save(images, mask>0.5, gt=None, cmap=cmap, save_path=save_path +'/tp', save_name=f'{id}_{d}')
        
        # fp (by classification model)
        elif (data["label"][0] == 0) and (predict_pos==True):
            mask = inferer_seg(images)
            d = di(mask, gt)
            
            # tn
            if (mask>0.5).sum()<seg_threshold:
                prob = prob*0
                save(images, dummy*0, gt=None, save_path=save_path +'/tn', save_name=f'{id}_{d}')
            
            # fp
            else:
                fp_list.append(id)
                save(images, mask>0.5, gt=None, cmap=cmap, save_path=save_path +'/fp', save_name=f'{id}_{d}')

        # fn
        elif (data["label"][0] == 1) and (predict_pos==False):
            save(images, dummy*0, gt=None, cmap=cmap, save_path=save_path +'/fn', save_name=f'{id}')
        
        # tn
        elif (data["label"][0] == 0) and (predict_pos==False):
            save(images, dummy*0, gt=None, save_path=save_path +'/tn', save_name=f'{id}')
            pass

        # classification
        y.append(label.numpy().astype('int'))
        y_pred.append(prob.cpu().numpy())
        cls_size.append(lesion_size)



    #dice_result = np.array([tp_dice, tp_dice_size])
    dice_total = np.array([dice_list['dice'], dice_list['size']])
    
    """
    with open("tp_list_", "wb") as dd:
        pickle.dump(tp_id, dd)
    
    with open("fp_list", "wb") as dd:
        pickle.dump(fp_list, dd)
    """

    y = np.hstack(y).squeeze(0)
    y_pred = np.hstack(y_pred).squeeze(0)
    cls_size = np.array(cls_size)

    out = np.stack((cls_size, y, y_pred))

    #with open(f"dice_result_{describ}", "wb") as fp:
    #    pickle.dump(dice_result, fp)

    with open(f"y_p_{describ}", "wb") as fp:
        pickle.dump(out, fp)
    
    with open(f"dice_total_{describ}", "wb") as fp:
        pickle.dump(dice_total, fp)