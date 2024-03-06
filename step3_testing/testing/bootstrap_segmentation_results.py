import numpy as np
import pickle

n_bootstraps = 10000
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

def booting_seg(n_bootstraps, y_pred, des):
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred), len(y_pred))

        score = (y_pred[indices]).mean()
        bootstrapped_scores.append(score)
    
    stat(bootstrapped_scores, des)




def stat_seg(name=""):

    with open(f"dice_total_{name}", "rb") as fp:
        dice = pickle.load(fp)
        print(dice.shape)


    size = dice[1,:]
    select_idx_s = np.logical_and((size>0), (size< 0.032))
    select_idx_m = np.logical_and((size>=0.032), (size<0.080))
    select_idx_l = (size>=0.080)



    for size_thres, des in zip([select_idx_s, select_idx_m, select_idx_l],['dice_s','dice_m','dice_l']):
        selected = dice[0,:][size_thres]
        booting_seg(n_bootstraps, selected, des)

    booting_seg(n_bootstraps, dice[0,:], des='dice_all')
    print(f'\n\n')