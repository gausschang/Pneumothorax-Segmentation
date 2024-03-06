#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
#import json
import pickle
import numpy as np

for i in ['result_vit', 'result_eff']:
    with open(i, "rb") as fp:
        data = pickle.load(fp)

    fpr, tpr, thresholds = roc_curve(data[0,:], data[1,:])
    idx = np.argmax(tpr - fpr)
    print(f'{i} : ', thresholds[idx])