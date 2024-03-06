import os
import pandas as pd
import numpy as np
import json
import cv2


# original label folder
label_dir = ''

# save folder
save_dir = ''

json_dict = {}
for i in os.listdir(label_dir):
    file = os.listdir(os.path.join(label_dir,i))
    if len(file) > 0:
        json_dict[i] = os.path.join(label_dir,i,file[0])


mask_dict = {}
for k in json_dict:
    
    with open(json_dict[k]) as f:
        data = json.load(f)
        id = k
        mask_dict[id] = []
    
    # initialz mask
    mask = np.zeros([data['size'][1], data['size'][0]],dtype=np.int8)
    
    # read bboxes in json, one box at a time
    for shape in data['shapes']:
        if 'Rectangle' in shape.values():
            
            # read bbox coordination in integer for matrix index
            x_a, y_a = round(shape['a'][0]), round(shape['a'][1])
            x_b, y_b = round(shape['b'][0]), round(shape['b'][1])
            
            x = min(x_a, x_b)
            y = min(y_a, y_b)
            h = max(y_a, y_b) - min(y_a, y_b)
            w = max(x_a, x_b) - min(x_a, x_b)
            
            # masking
            mask[y:y+h,x:x+w] = 1
            
        else:
            pass

    mask_dict[id].append(mask)

for i in mask_dict:
    mask = mask_dict[i][0]
    mask = mask*255
    cv2.imwrite(os.path.join(save_dir,"{i}.png"), mask)
