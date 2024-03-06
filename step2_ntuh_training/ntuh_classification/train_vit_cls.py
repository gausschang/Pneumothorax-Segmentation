from logging import raiseExceptions
import warnings
warnings.filterwarnings('ignore')


import os
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
torch.multiprocessing.set_sharing_strategy('file_system')

import monai
from monai.data import list_data_collate

from torch.utils.tensorboard import SummaryWriter


from tinyvit_cls import TinyViT
from data import *
from utils import *
from load import *

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

monai.utils.misc.set_determinism(seed=0, use_deterministic_algorithms=True)
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'

def train(max_epoch = 100, batch_size=8, num_samples=5000, fold_num=0, pt_path=''):

    project_name = f'fold_{fold_num}_vit_cls_ntuh'
    save_dir = f'./model_weights/{project_name}/'
    writer = SummaryWriter(log_dir=f'runs/{project_name}', comment=f"{project_name}")
    
    lr = 1e-4/10
    bs = batch_size

    # get transform
    fixed_train_transforms, val_transforms = get_transform(input_size=512)

    # prep dataset
    ntuh_data_list, _ = PneuDataset()

    train_pos, val_pos = get_5fold(ntuh_data_list, k=fold_num)
    train_neg, val_neg = get_5fold(ntuh_empty(), k=fold_num)

    train_list = train_pos + train_neg
    val_list = val_pos + val_neg

    val_ds = monai.data.Dataset(data=val_list, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=0, collate_fn=list_data_collate)
    
    train_ds = monai.data.Dataset(data=train_list, transform=fixed_train_transforms)


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
    model = load(model, pt_path)
    model = nn.DataParallel(model)
    model.to(device)

    model_size = count_parameters(model)
    print(f'# of trainable parameter : {model_size}')

    criterion = monai.losses.DiceLoss(to_onehot_y=True,softmax=True)
    bce = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr , weight_decay=0.05)
    
    #-----------------------------start training -------------------

    iter_count = 0
    best_auc = 0
    best_dice = 0
    

    for epoch in range(max_epoch):
        sample_weight = update_w(train_list, epoch, max_epoch)
        sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weight, num_samples=num_samples, replacement=True) ###############

        train_loader = monai.data.DataLoader(
                train_ds,
                batch_size=bs,
                num_workers=0,
                sampler=sampler,
                collate_fn=list_data_collate,
                drop_last=True,
                #pin_memory=True
                )

        print(f"-------------------{project_name}--------------------------")
        print(f"epoch {epoch + 1}/{max_epoch}")

        model.train()

        epoch_loss = 0
        epoch_loss_valid = 0
        step = 0

        for batch_data in tqdm.tqdm(train_loader):

            iter_count += 1
            warmup_cos_iter(optimizer, epoch_iter=len(train_loader), current_iter=iter_count, max_iter=max_epoch*len(train_loader),lr_min=1e-8, lr_max=lr, warmup=True)

            step += 1
            
            inputs = batch_data["image"].to(device, dtype=torch.float)
            masks = batch_data["mask"].to(device, dtype=torch.float)
            label = batch_data["label"].to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs, aux = model(inputs)

            loss = 0.5*criterion(outputs, masks) + bce(aux, label)

            loss.mean().backward()
            _ = grad_norm(model, writer, step=iter_count)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_loss /= step
        print(f"training_loss={epoch_loss:.4f}")

        if True:
            model.eval()
            with torch.no_grad():

                auc = 0
                ntuh = []

                y_pred=[]
                y=[]

                for val_data in tqdm.tqdm(val_loader):

                    val_images = val_data["image"].to(device, dtype=torch.float)
                    val_masks = val_data["mask"].to(device, dtype=torch.float)
                    val_label = val_data['label'].to(device, dtype=torch.float)
                    
                    pred, aux = model(val_images)

                    loss_valid = 0.5*criterion(pred, val_masks) + bce(aux, val_label)

                    epoch_loss_valid += loss_valid.mean().item()/len(val_loader)

                    # evaluate dice score
                    if val_data["label"][0] == 1:
                        pred = torch.nn.functional.softmax(pred, dim=1)
                        binary_outputs_masks = (pred[:,1,:,:] >= 0.5).unsqueeze(dim=1)
                        one_score = monai.metrics.compute_meandice(binary_outputs_masks, val_masks).cpu().item()
                        ntuh.append(one_score)

                    # evaluate cls
                    p = torch.sigmoid(aux)
                    y.append(val_label.cpu().numpy().astype('int'))
                    y_pred.append(p.cpu().numpy())

                y = np.hstack(y).squeeze(0)
                y_pred = np.hstack(y_pred).squeeze(0)
                auc = roc_auc_score(y,y_pred, average=None)
                
                dice_val = np.array(ntuh).mean()

                print(f"val_loss={epoch_loss_valid:.4f}")


                if dice_val > best_dice:
                    best_dice = dice_val
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    #torch.save(model.state_dict(), os.path.join(save_dir, f"{project_name}_dice"))

                if auc > best_auc:
                    best_auc = auc
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    torch.save(model.state_dict(), os.path.join(save_dir, f"{project_name}"))


                # log
                writer.add_scalars(f'Loss', {'train':epoch_loss,'val':epoch_loss_valid}, epoch)
                writer.add_scalars('Val_Dice', {'ntuh':dice_val}, epoch)
                writer.add_scalars('AUC', {'auc':auc}, epoch)

    print(f"train completed")
    print(f'# of trainable parameter : {model_size}')
    os.rename(os.path.join(save_dir, f"{project_name}"), os.path.join(save_dir, f"{project_name}_"))
    #os.rename(os.path.join(save_dir, f"{project_name}_dice"), os.path.join(save_dir, f"{project_name}_dice{best_dice}"))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    #main()