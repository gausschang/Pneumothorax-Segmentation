import monai
import pickle
from get_data import *
from sys_inference import *
import json

# Opening JSON file
js_path = '../../step2_ntuh_training/get_thres/cls_vit_5fold_result/5cv_avg.json'
if os.path.isfile(js_path):
    with open(js_path) as json_file:
        data = json.load(json_file)
        # Print the data of dictionary
        youden = float(data['cls_threshold'])
        seg_threshold = float(data['seg_threshold'])
else:
    raise ValueError(f"threshold value file not exist at : {js_path}")

cls_weights = [f'../../step2_ntuh_training/ntuh_classification/model_weights/fold_{i}_vit_cls_ntuh/fold_{i}_vit_cls_ntuh_' for i in range(5)]
seg_weights = [f'../../step2_ntuh_training/ntuh_segmentation/model_weights/fold_{i}_vit_seg_ntuh/fold_{i}_vit_seg_ntuh_' for i in range(5)]

seg_config = {'encoder_name':None, 'weight_list':seg_weights}
cls_config = {'encoder_name':None, 'weight_list':cls_weights}
name = 'vit'
cmap = 'spring_r'

def main(description, cls_config, seg_config, youden, seg_threshold, cmap, save_dir):

    val_list = testDataset()

    val_ds = monai.data.Dataset(data=val_list, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=monai.data.list_data_collate)

    predict(val_loader, cls_config, seg_config, youden, seg_threshold, cmap, save_dir, description)

    
main(name, cls_config, seg_config, youden, seg_threshold, cmap, save_dir=f'/data2/smarted/PXR/data/C426_Pneumothorax_preprocessed/2023_project_handover_ptx/{name}')

