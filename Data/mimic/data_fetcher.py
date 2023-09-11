import numpy as np
import torch
import os
import pandas as pd
from sqlalchemy import create_engine
import configparser
from tqdm import tqdm
from Data.mimic.mimic import get_events, get_all_icustay_ids
import dill
import json
from settings import ROOT_DIR


mimic_path = os.path.join(ROOT_DIR, 'Data', 'mimic')
chart_dict = json.load(open(os.path.join(mimic_path, 'chartids.json'), 'r'))
lab_dict = json.load(open(os.path.join(mimic_path, 'labids.json'), 'r'))


def load_initialized_dataset(n_steps: int, time_step: int)->torch.utils.data.Dataset:
    """ load time series data class if it was saved before

    Parameters
    ----------

    n_steps:
        number of timesteps

    time_step:
        length of timesteps in hours
    """
    with open(f'dataset_{n_steps}_{time_step}.p', 'rb') as f:
        data = dill.load(f)
    return data

def return_ids(itemid_dict):
    all_ids = []
    for ids in itemid_dict.values():
        for id in ids:
            all_ids.append(id)
    return tuple(all_ids)


def map_item_ids(chart_dict: dict) -> dict:
    replace_dict = {}
    for i, values in enumerate(chart_dict.values()):
        for id in values:
            replace_dict[id] = i
    return replace_dict

def charttime_to_timesteps(df, time_step):
    df.loc[:, 'charttime'] = df['charttime'] - df['charttime'].min()
    charttime_hours = df.charttime/pd.Timedelta(hours=1)
    charttime_step = charttime_hours/time_step
    df.loc[:, 'charttime'] = charttime_step
    return df



def dataframe_to_array(df: pd.DataFrame, n_steps: int, time_step: int, n_features: int):

    ids = df.icustay_id.unique()
    n_samples = len(ids)
    print(n_samples)
    data = np.zeros((n_samples, n_features, n_steps))
    icustay_group = df.groupby(['icustay_id'])

    for i, (name, group) in tqdm(enumerate(icustay_group)):
        temp_df = group.copy()
        temp_df = charttime_to_timesteps(temp_df, time_step)
        for step in range(n_steps): #very slow
            for feature in range(n_features):
                v_ = temp_df[(temp_df['itemid']==feature) & (temp_df['charttime'].between(step*time_step, (step+1) * time_step))]['valuenum']
                v = np.mean(v_)
                data[i, feature, step] = v
    return data, ids




def make_dataframe(engine, chart_dict, lab_dict, icustay_id):
    chart_ids = return_ids(chart_dict)
    lab_ids = return_ids(lab_dict)
    with engine.connect() as connection:
        df = get_events(connection, chart_ids, lab_ids, icustay_id).dropna().drop('hadm_id', axis=1).astype({'icustay_id': int})
    all_dict = dict(list(chart_dict.items()) + list(lab_dict.items()))
    replace_dict = map_item_ids(all_dict)
    df = df.replace({'itemid': replace_dict}).drop_duplicates()
    return df.drop_duplicates()


def alchemy_engine_mimic(config_path):
    config = configparser.ConfigParser()
    config.read(config_path)
    alchemy_config = config['sqlalchemy']
    url = alchemy_config['url']
    engine = create_engine(url, connect_args={'options': '-csearch_path={}'.format('mimiciii')})

    return engine

class TimeSeriesData(torch.utils.data.Dataset):
    """ Time Series Data Class. Requires mimic3 database

    Parameters
    ----------
    chart_dict:
        dict with key as item labels and values as a list of chart ids in mimic3 database

    lab_dict:
        dict with key as item labels and values as a list of  ab ids in mimic3 database

    n_steps:
        number of timesteps

    time_step:
        length of timesteps in hours
    """

    def __init__(self, chart_dict: dict, lab_dict: dict, n_steps: int, time_step: int, use_lab=True, heart_only=False):
        engine = alchemy_engine_mimic('mimic.ini')
        self.chart_dict = chart_dict
        self.lab_dict = lab_dict
        self.icustay_ids = get_all_icustay_ids(engine, heart_only=heart_only)
        print(len(self.icustay_ids))
        self.n_steps = n_steps

        self.time_step = time_step
        self.n_features = len(self.chart_dict)
        if use_lab:
            self.n_features += len(self.lab_dict)
        self.ts_array_all_ids, self.array_ids = self.make_array_all_ids(engine)
        del engine

    def __len__(self):
        return len(self.ts_array_all_ids)


    def __getitem__(self, idx):
        return self.ts_array_all_ids[idx]

    def make_array_all_ids(self, engine):
        print('load data from mimic')
        df = make_dataframe(engine, self.chart_dict, self.lab_dict, self.icustay_ids)
        print('create array')
        ts_array, ids = dataframe_to_array(df, self.n_steps, self.time_step, self.n_features)
        return ts_array, ids

if __name__ == '__main__':
    n_steps = 48
    time_step = 1
    # data = TimeSeriesData(chart_dict,lab_dict, n_steps, time_step, heart_only=False)
    # print('fertig')
    # with open(f'dataset_{n_steps}_{time_step}_2510_heart.p', 'wb') as f:
    #     dill.dump(data, f)
    # arr = data.ts_array_all_ids
    # ids = data.array_ids
    # np.save(f'dataset_{n_steps}_{time_step}_29_03_arr_all.npy', arr)
    # np.save(f'ids_all_data', ids)
    # data = load_initialized_dataset(48, 1)
    engine = alchemy_engine_mimic('/home/steven/PycharmProjects/data-imputation-icu/Data/mimic/mimic.ini')
    heart_ids = get_all_icustay_ids(engine, heart_only=True)
    # stays = get_all_icustay_ids(engine)[:10]
    # n_features = len(chart_dict) + len(lab_dict)
    # df = make_dataframe(engine, chart_dict, lab_dict, stays)
    # x = dataframe_to_array(df, 48, 1, n_features)
