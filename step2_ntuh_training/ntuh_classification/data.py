import os
import torch
import monai
from monai.transforms import (
    NormalizeIntensityd,
    AddChanneld,
    AsChannelFirstd,
    Compose,
    LoadImaged,
    RandZoomd,
    RandRotated,
    RandFlipd,
    EnsureTyped,
    Resized,
    Lambdad,
    RandSpatialCropd,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandAdjustContrastd,
)
from math import pi
import pandas as pd


def PneuDataset():
    
    data_dir = "/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data"
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

def get_5fold(data_list, k=0):
    n = int(0.2*len(data_list))
    if k==4:
        val_list = data_list[k*n : ]
    else:
        val_list = data_list[k*n : (k+1)*n]
    train_list = [data for data in data_list if data not in val_list]
    return train_list, val_list


def get_transform(input_size=512):
    fixed_train_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            AsChannelFirstd(keys=["image", "mask"]),
            AddChanneld(keys=["image", "mask"]),
            Resized(keys=["image","mask"], spatial_size=(1024,1024)),
            monai.transforms.RepeatChanneld(keys=["image"], repeats=3, allow_missing_keys=False),
            
            RandZoomd(keys=["image", "mask"], min_zoom=0.8, max_zoom=1.2, mode="area", padding_mode="constant", prob=0.2),
            RandSpatialCropd(keys=["image", "mask"],
                    roi_size=(int(input_size/0.8),int(input_size/0.8)), random_center=True, random_size=True),
            Resized(keys=["image","mask"], spatial_size=(input_size,input_size)),

            RandGaussianSmoothd(keys=["image"], sigma_x=(0.5,1.15), sigma_y=(0.5,1.15), prob=0.15),
            RandScaleIntensityd(keys=["image"], factors=0.3, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.3), prob=0.5),
            RandGaussianNoised(keys=["image"], std=0.8, prob=0.15),
            
            RandRotated(keys=["image", "mask"], range_x=round(1.5*pi/18, 3), prob=0.5, padding_mode="zeros"),
            RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=-1),

            NormalizeIntensityd(keys=["image"], channel_wise=True, subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
            Lambdad(keys='mask', func=lambda x: torch.from_numpy(x > 127).long() ),
            EnsureTyped(keys=["image"], dtype=torch.float)
            ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        AsChannelFirstd(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        monai.transforms.Resized(keys=["image","mask"], spatial_size=(input_size,input_size)),
        monai.transforms.RepeatChanneld(keys=["image"], repeats=3, allow_missing_keys=False),
        NormalizeIntensityd(keys=["image"], channel_wise=True, subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
        Lambdad(keys='mask', func=lambda x: torch.from_numpy(x > 127).long() ),
        EnsureTyped(keys=["image"], dtype=torch.float)
        ])

    return fixed_train_transforms.set_random_state(0), val_transforms

def nih_empty():
    
    img_dir = "/data2/smarted/PXR/data/0_NTUH_PNEUMOTHORAX/images_dcm2png"
    mix_mask_dir = '/data2/smarted/PXR/data/siim-acr-pneumothorax/png_masks'
    mask_dir = "/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/siim/mask"

    mix = os.listdir(img_dir)
    pos = os.listdir(mask_dir)
    mix.sort()

    empty_list = [i for i in mix if ((i not in pos) and (i[0]!='P'))]
    empty_list.sort()

    #empty_list = empty_list[:nih_neg_sample]
    if len(mix) == len(empty_list) :
        raise ValueError('something wrong')

    empty_data_dict = []
    for i in empty_list:
        em_mask_path = os.path.join(mix_mask_dir, i)
        empty_data_dict.append({"image": os.path.join(img_dir, i), 'mask': em_mask_path, "label":torch.tensor([0]), "Source": 'siim_e'})

    return empty_data_dict

def ntuh_empty():
    empty_data_dict=[]
    img_dir = "/data2/smarted/PXR/data/0_NTUH_PNEUMOTHORAX/images_dcm2png"
    empty_dir = '/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/ntuh_all_neg'
    l = os.listdir(empty_dir)
    l.sort()
    for i in l:
        empty_data_dict.append({"image": os.path.join(img_dir, i), 'mask': os.path.join(empty_dir, i), "label":torch.tensor([0]), "Source": 'ntuh_e'})
    return empty_data_dict

def update_w(train_list, epoch, max_epoch):
    p = 0.8 - 0.5*(epoch/max_epoch)
    l = []
    for i in train_list:
        if i['label']==0:
            l.append((1-p)/9000)
        elif i['label']==1:
            l.append(p/2700)
    w_tensor = torch.FloatTensor(l)
    return w_tensor