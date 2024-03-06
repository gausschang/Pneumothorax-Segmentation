from bootstrap_classification_results import *
from bootstrap_segmentation_results import *
import os
import json

# Opening JSON file
js_path = '../../step2_ntuh_training/get_thres/cls_eff_5fold_result/5cv_avg.json'
if os.path.isfile(js_path):
    with open(js_path) as json_file:
        data = json.load(json_file)
        # Print the data of dictionary
        youden = float(data['cls_threshold'])
else:
    raise ValueError(f"threshold value file not exist at : {js_path}")

print('=================start of cnn model report=================')
stat_seg(name="eff")
stat_cls(name="eff", threshold=youden)
print('=================end of cnn model report===================\n\n\n')


# Opening JSON file
js_path = '../../step2_ntuh_training/get_thres/cls_vit_5fold_result/5cv_avg.json'
if os.path.isfile(js_path):
    with open(js_path) as json_file:
        data = json.load(json_file)
        # Print the data of dictionary
        youden = float(data['cls_threshold'])
else:
    raise ValueError(f"threshold value file not exist at : {js_path}")

print('=================start of vit model report=================')
stat_seg(name="vit")
stat_cls(name="vit", threshold=youden)
print('=================end of cnn model report===================')