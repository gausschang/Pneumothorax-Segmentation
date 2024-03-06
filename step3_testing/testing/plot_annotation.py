

import tqdm
import numpy as np
import cv2
import os
import monai
from get_data import *

def draw(save_dir, mode='gt'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    val_transforms = val_transforms = Compose([
            LoadImaged(keys=["image", "mask"]),
            Lambdad(keys='image', func=lambda x: x[...,2]),
            AsChannelFirstd(keys=["image", "mask"]),
            AddChanneld(keys=["image", "mask"]),
            Resized(keys=["image","mask"], spatial_size=(512,512)),
            EnsureTyped(keys=["image","mask"], dtype=torch.float)
            ])

    val_list = testDataset()
    val_ds = monai.data.Dataset(data=val_list, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=monai.data.list_data_collate)

    for batch_data in tqdm.tqdm(val_loader):

        image = batch_data["image"][0,0,:,:].numpy().astype('uint8')
        mask = batch_data["mask"][0,0,:,:].numpy().astype('uint8')
        id = batch_data["id"][0]

        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        if mode == 'gt':
            # convert mask_png to binary
            t, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (102,220,0), 2, lineType=cv2.LINE_4, offset = (0,0))
        else:
            pass

        cv2.imwrite(save_dir + id + '.png', image)

save_dir = '/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/ptx_testing_gt/'
draw(save_dir, '')