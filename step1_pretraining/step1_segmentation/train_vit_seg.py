from logging import raiseExceptions
import warnings
warnings.filterwarnings('ignore')


import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')

import monai
from monai.losses import DiceLoss
from monai.data import list_data_collate

import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

from tinyvit_seg import TinyViT
from data import *
from utils import *
from load import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

monai.utils.misc.set_determinism(seed=0, use_deterministic_algorithms=True)
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

def train(max_epoch = 100, input_size=512):

    project_name = f'tinyvit_siim'
    save_dir = f'./model_weights/{project_name}/'
    writer = SummaryWriter(log_dir=f'runs/{project_name}', comment=f"{project_name}")
    
    lr = 1e-4
    bs = 4

    # get transform
    fixed_train_transforms, val_transforms = get_transform(input_size=input_size)

    # prep dataset
    _, siim_data_list = PneuDataset()

    siim_train, siim_val = siim_data_list[:-200], siim_data_list[-200:]


    val_ds = monai.data.Dataset(data=siim_val, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
    
    train_ds = monai.data.Dataset(data=siim_train, transform=fixed_train_transforms)
    train_loader = monai.data.DataLoader(
            train_ds,
            batch_size=bs,
            num_workers=4,
            collate_fn=list_data_collate,
            shuffle=True,
            drop_last=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ########################## model ###########################
    model = TinyViT(
                img_size=512,
                num_classes=2,
                embed_dims=[96, 192, 384, 576],
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 18],
                window_sizes=[16, 16, 32, 16],
                drop_path_rate=0.1,
            )
    model = load(model, './tiny_vit_21m_22kto1k_512_distill.pth') # ImageNet pretrain weight for TinyViT
    model.to(device)

    model_size = count_parameters(model)

    print(f'# of trainable parameter : {model_size}')

    criterion = DiceLoss(to_onehot_y=True,softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    
    #-----------------------------start training -------------------

    iter_count = 0
    best_score = 0

    for epoch in range(max_epoch):
        print(f"-------------------{project_name}--------------------------")
        print(f"epoch {epoch + 1}/{max_epoch}")

        model.train()

        epoch_loss = 0
        epoch_loss_valid = 0
        step = 0

        for batch_data in tqdm.tqdm(train_loader):

            iter_count += 1
            warmup_cos_iter(optimizer, epoch_iter=len(train_loader), current_iter=iter_count, max_iter=max_epoch*len(train_loader),lr_min=1e-7, lr_max=lr, warmup=True)

            step += 1
            
            inputs = batch_data["image"].to(device, dtype=torch.float)
            masks = batch_data["mask"].to(device, dtype=torch.float)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, masks)


            loss.backward()
            _ = grad_norm(model, writer, step=iter_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= step
        print(f"training_loss={epoch_loss:.4f}")

        if True:
            model.eval()
            with torch.no_grad():

                score = 0
                siim = []
                ntuh = []

                for val_data in tqdm.tqdm(val_loader):

                    val_images = val_data["image"].to(device, dtype=torch.float)
                    val_masks = val_data["mask"].to(device, dtype=torch.float)
                    
                    pred = model(val_images)

                    loss_valid = criterion(pred, val_masks)

                    epoch_loss_valid += loss_valid.item()/len(val_loader)

                    # evaluate dice score
                    pred = torch.nn.functional.softmax(pred, dim=1)
                    binary_outputs_masks = (pred[:,1,:,:] >= 0.5).unsqueeze(dim=1)
                    a_score = monai.metrics.compute_meandice(binary_outputs_masks, val_masks).cpu().item()
                    score += a_score/len(val_loader)

                    siim.append(a_score)
                
                siim = np.array(siim).mean()
                score = siim
                print(f"val_loss={epoch_loss_valid:.4f}")


                if score > best_score:
                    best_score = score
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{project_name}"))


                # log
                writer.add_scalars(f'Loss', {'train':epoch_loss,'val':epoch_loss_valid}, epoch)
                writer.add_scalars('Val_Dice', {'siim':siim}, epoch)

    print(f"train completed")
    print(f'# of trainable parameter : {model_size}')
    os.rename(os.path.join(save_dir, f"{project_name}"), os.path.join(save_dir, f"{project_name}_"))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    #main()