import os
from queue import Empty
import torch
import pandas as pd
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

def testDataset():
    
    total_cases = '/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/labels_G301'
    mask_dir = '/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/mask_G301'
    img_dir = '/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/image/preprocessed/images_G301'

    test_data_list = []
    
    for i in os.listdir(total_cases):

        image_path = os.path.join(img_dir, i+'.png')
        label_path = os.path.join(total_cases, i, 'label.json')
        if os.path.exists(label_path):
            label = torch.tensor([1])
            mask_path = os.path.join(mask_dir, i+'.png')
        else:
            label = torch.tensor([0])
            mask_path = '/data2/smarted/PXR/data/siim-acr-pneumothorax/work/data/ntuh_all_neg/P214260000003.png'

        test_data_list.append({"image": image_path, "mask": mask_path, "label":label, "id": i})

    return test_data_list



val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        Lambdad(keys='image', func=lambda x: x[...,2]),
        AsChannelFirstd(keys=["image", "mask"]),
        AddChanneld(keys=["image", "mask"]),
        Resized(keys=["image","mask"], spatial_size=(512,512)),
        RepeatChanneld(keys=["image"], repeats=3, allow_missing_keys=False),
        NormalizeIntensityd(keys=["image"], channel_wise=True, subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
        Lambdad(keys='mask', func=lambda x: x > 127),
        EnsureTyped(keys=["image","mask"], dtype=torch.float)
        ])


