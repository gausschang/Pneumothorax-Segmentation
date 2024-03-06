from data import *

def fold_stat(fold_num):

    ntuh_data_list, siim_data_list = PneuDataset()

    train_list, val_list = get_5fold(ntuh_data_list, k=fold_num)
    empty_train, empty_val = get_5fold(ntuh_empty(), k=fold_num)


    tt = len(val_list)+len(empty_val)
    print(f'--------------fold {fold_num}---------------')
    print(f'total = {tt}')
    print(f'pos = {len(val_list)}')
    print(f'pos rate = {len(val_list)/tt}')
    return None

for i in range(5):
    fold_stat(i)