

import os
import glob
import pydicom
import numpy as np
import cv2
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
from monai.data import (
    DataLoader,
    Dataset,
    list_data_collate
    )


def read_dcm(
    dcm_path,
    fix_monochrome=True,
    normalization=True,
    apply_window=True,
    range_correct=True
    ):
    dicom = pydicom.read_file(dcm_path)
    # For ignoring the UserWarning: "Bits Stored" value (14-bit)...
    #uid = dicom[0x0008, 0x0018].value
    uid = os.path.basename(dcm_path.replace('\\','/')).replace('.dcm','')
    elem = dicom[0x0028, 0x0101]
    elem.value = 16

    data = dicom.pixel_array

    median = np.median(data)
    if range_correct:
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.where(data == 0, median, data)
        else:
            data = np.where(data == 4095, median, data)

    if normalization:
        if apply_window and "WindowCenter" in dicom and "WindowWidth" in dicom:
            window_center = float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth)
            y_min = (window_center - 0.5 * window_width)
            y_max = (window_center + 0.5 * window_width)
        else:
            y_min = data.min()
            y_max = data.max()
        data = (data - y_min) / (y_max - y_min)
        data = np.clip(data, 0, 1)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = (data * 255.).astype(np.uint8)

    return data, uid


def get_loader(dcm_paths=[]):
    datalist = []

    temp_dir = './temp_dir/'
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    for i in dcm_paths:
        data, uid = read_dcm(
                        dcm_path=i,
                        fix_monochrome=True,
                        normalization=True,
                        apply_window=True,
                        range_correct=True
                    )
        image_path = os.path.join(temp_dir, f'{uid}.png')
        cv2.imwrite(image_path, data)
        datalist.append({'id':uid, 'image': image_path})

    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        AsChannelFirstd(keys=["image"]),
        AddChanneld(keys=["image"]),
        RepeatChanneld(keys=["image"], repeats=3, allow_missing_keys=False),
        Resized(keys=["image"], spatial_size=(512,512)),
        NormalizeIntensityd(keys=["image"], channel_wise=True, subtrahend=[0.485, 0.456, 0.406], divisor=[0.229, 0.224, 0.225]),
        EnsureTyped(keys=["image"], dtype=torch.float),
        ])
    test_ds = Dataset(data=datalist, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    return test_loader

