import os
import torch
from monai.transforms import (
    NormalizeIntensityd,
    AddChanneld,
    AsChannelFirstd,
    Compose,
    LoadImaged,
    EnsureTyped,
    Resized,
    RepeatChanneld,
    Lambdad,
)
from math import pi, cos
import pandas as pd

def PneuDataset():
    
    data_dir = "/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data"
    #metas = pd.read_csv('/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/dataset_meta.csv')
    metas = pd.read_csv('/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/dataset_meta_no_holdout.csv')
    metas = metas.sample(frac=1, random_state=0)

    ntuh_data_list = []
    siim_data_list = []
    
    for idx, row in metas.iterrows():
        image_path = os.path.join(data_dir, 'ntuh_all_positive' if row['Source']=='ntuh' else 'siim', 'img', row['File_Name'])
        mask_path = os.path.join(data_dir, 'ntuh_all_positive' if row['Source']=='ntuh' else 'siim', 'mask', row['File_Name'])
        view = row['ViewPosition']
        if row['Source']=='ntuh':
            ntuh_data_list.append({"image": image_path, "mask": mask_path, "Source": row['Source'], "label":torch.tensor([1])})
        if row['Source']=='siim':
            siim_data_list.append({"image": image_path, "mask": mask_path, "Source": row['Source'], "label":torch.tensor([1])})
    return ntuh_data_list, siim_data_list

def ntuh_empty():
    empty_data_dict=[]
    img_dir = "/data2/smarted/PXR/data/0_NTUH_PNEUMOTHORAX/images_dcm2png"
    #empty_dir = '/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/ntuh_empty_train'
    empty_dir = '/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/ntuh_all_neg'
    l = os.listdir(empty_dir)
    l.sort()
    for i in l:
        empty_data_dict.append({"image": os.path.join(img_dir, i), 'mask': os.path.join(empty_dir, i), "label":torch.tensor([0]), "Source": 'ntuh_e'})
    return empty_data_dict

"""
def get_5fold(data_list, k=0):
    n = int(0.2*len(data_list))
    val_list = data_list[k*n : (k+1)*n]
    train_list = [data for data in data_list if data not in val_list]
    return train_list, val_list
"""
def get_5fold(data_list, k=0):
    n = int(0.2*len(data_list))
    if k==4:
        val_list = data_list[k*n : ]
    else:
        val_list = data_list[k*n : (k+1)*n]
    train_list = [data for data in data_list if data not in val_list]
    return train_list, val_list

val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),
    AsChannelFirstd(keys=["image", "mask"]),
    AddChanneld(keys=["image", "mask"]),
    RepeatChanneld(keys=["image"], repeats=3, allow_missing_keys=False),
    Resized(keys=["image","mask"], spatial_size=(512,512)),
    NormalizeIntensityd(keys=["image"], channel_wise=True, subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
    Lambdad(keys='mask', func=lambda x: (x > 127) ),
    EnsureTyped(keys=["image", "mask"], dtype=torch.float),
    ])


