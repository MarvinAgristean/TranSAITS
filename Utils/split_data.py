import numpy as np
from Data.mimic.mimic import get_all_icustay_ids
import math

def list_in(a, b):
    return any(map(lambda x: b[x:x + len(a)] == a, range(len(b) - len(a) + 1)))

def create_sets(arr_path, all_id_path, heart_id_path, split_sizes):
    assert sum(split_sizes) == 1
    arr = np.load(arr_path)
    all_ids=np.load(all_id_path)
    heart_ids = np.load(heart_id_path)
    train_size, test_size, val_size = split_sizes

    n_stays = len(arr)
    n_test = math.floor(n_stays * test_size)
    n_val = math.floor(n_stays * val_size)

    test_ids = np.random.choice(heart_ids, size=n_test, replace=False)
    temp_ids_ = all_ids[~np.isin(all_ids, test_ids)]
    

    val_ids = np.random.choice(heart_ids, size=n_val, replace=False)
    
    train_ids = temp_ids_[~np.isin(temp_ids_, val_ids)]
    
    train_arr = arr[np.isin(all_ids, train_ids)]
    test_arr = arr[np.isin(all_ids, test_ids)]
    val_arr = arr[np.isin(all_ids, val_ids)]
    return train_arr, test_arr, val_arr


if __name__ == '__main__':
    train, test,val = create_sets('/home/steven/PycharmProjects/data-imputation-icu/Data/mimic/dataset_48_1_29_03_arr_all.npy',
                                  '/home/steven/PycharmProjects/data-imputation-icu/Data/mimic/ids_all_data.npy',
                                  '/home/steven/PycharmProjects/data-imputation-icu/Data/mimic/ids_heart_stays.npy',
                                  [0.6,0.2,0.2])
    data = {'train': train, 'test': test, 'val': val}