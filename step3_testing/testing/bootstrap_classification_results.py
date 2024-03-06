import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, average_precision_score
import pickle

n_bootstraps = 1000
rng_seed = 0  # control reproducibility
rng = np.random.RandomState(rng_seed)

def stat(bootstrapped, metric_name):
    sorted_scores = np.array(bootstrapped)
    sorted_scores.sort()

    mean = sorted_scores.mean()

    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    print(f'{metric_name} :')
    print(f"{mean:.3f} ({confidence_lower:.3f}-{confidence_upper:.3f})")
    print('-------------------------')

def booting(n_bootstraps, idx, y_true, y_pred, threshold, des):

    print('data_size:', idx.sum())
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    bootstrapped_auc = []
    bootstrapped_auprc = []
    bootstrapped_sen = []
    bootstrapped_spe = []
    bootstrapped_ppv = []
    bootstrapped_npv = []

    for i in range(n_bootstraps):
        print(f'{i}-th iteration', end='\r')
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))
        true, pred = y_true[indices], y_pred[indices]

        auc = roc_auc_score(true, pred)
        bootstrapped_auc.append(auc)

        auprc = average_precision_score(true, pred)
        bootstrapped_auprc.append(auprc)

        tn, fp, fn, tp = confusion_matrix(true, pred>threshold).ravel()

        ppv = tp/(tp+fp)
        npv = tn/(tn+fn)
        sen = tp/(tp+fn)
        spe = tn/(tn+fp)

        bootstrapped_sen.append(sen)
        bootstrapped_spe.append(spe)
        bootstrapped_npv.append(npv)
        bootstrapped_ppv.append(ppv)

    print(f'-------------lesion: {des}-----------------')
    
    stat(bootstrapped_sen, 'sensitivity')
    stat(bootstrapped_spe, 'spec')
    stat(bootstrapped_ppv, 'ppv')
    stat(bootstrapped_npv, 'npv')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred>threshold).ravel()
    cm_txt = np.array([['TP','FP'], ['FN','TN']])
    cm = np.array([[tp,fp], [fn,tn]])
    print(f'confusion matrix :\n{cm_txt} \n = \n {cm}')

    #stat(bootstrapped_auprc, 'auprc')
    #stat(bootstrapped_auc, 'auc')
    print(f'\n\n')


def stat_cls(name='', threshold=0):
    with open(f"y_p_{name}", "rb") as fp:
        data = pickle.load(fp)

    size = data[0,:]
    y_true = data[1,:]
    y_pred = data[2,:]

    select_idx_s = np.logical_and((size>0), (size< 0.032))
    select_idx_m = np.logical_and((size>=0.032), (size<0.080))
    select_idx_l = (size>=0.080)



    booting(n_bootstraps, size>-1, y_true, y_pred, threshold, des='all')
    for select_idx, des in zip([select_idx_l, select_idx_m, select_idx_s],['l','m','s']):
        select_idx = np.logical_or(select_idx, size==0) # add negative sample
        booting(n_bootstraps, select_idx, y_true, y_pred, threshold, des)
    