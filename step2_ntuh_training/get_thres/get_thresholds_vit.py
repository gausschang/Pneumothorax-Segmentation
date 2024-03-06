import pickle
import monai
from data import *
from predict import *
from save_result import auc

cls_weights = [f'../ntuh_classification/model_weights/fold_{i}_vit_cls_ntuh/fold_{i}_vit_cls_ntuh_' for i in range(5)]
seg_weights = [f'../ntuh_segmentation/model_weights/fold_{i}_vit_seg_ntuh/fold_{i}_vit_seg_ntuh_' for i in range(5)]


def main(fold_num, cls_config, seg_config, save_dir):

    ntuh_data_list, siim_data_list = PneuDataset()

    train_list, val_list = get_5fold(ntuh_data_list, k=fold_num)
    empty_train, empty_val = get_5fold(ntuh_empty(), k=fold_num)

    #print(len(val_list))
    #print(len(empty_val))

    val_list = val_list + empty_val

    val_ds = monai.data.Dataset(data=val_list, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=monai.data.list_data_collate)

    yall = predict(val_loader, cls_config, seg_config, save_dir, save_info=fold_num)
    return yall


y_out = []
y_prob = []
size_threshold = []
for fold_num in range(5):
    cls_config = {'encoder_name':None, 'weight_list':[cls_weights[fold_num]]}
    seg_config = {'encoder_name':None, 'weight_list':[seg_weights[fold_num]]}

    y, y_pred, detectable = main(fold_num, cls_config, seg_config, save_dir=f'cls_vit_5fold_result')
    y_out.append(y)
    y_prob.append(y_pred)
    size_threshold.append(detectable)

size_threshold = np.array(size_threshold).min()
print('segmentation_threshold',size_threshold)
y_out = np.hstack(y_out)
p_out = np.hstack(y_prob)

out = np.stack((y_out,p_out))

auc(out[0,:],out[1,:], size_threshold, save_dir='cls_vit_5fold_result', save_name='5cv_avg')
#with open("result_vit.json", "w") as fp:
#    json.dump(str(out), fp)
