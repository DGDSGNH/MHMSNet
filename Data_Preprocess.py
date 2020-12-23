import pandas as pd
import numpy as np
import torch
from torch.nn import utils as nn_utils
import pickle

def csv_to_fill_mimic3():
    csv_data = pd.read_csv("lb_mimic3.csv", low_memory=False)
    csv_data = pd.concat([csv_data[['reporttime', 'hadm_id', 'gender', 'hospital_expire_flag']],
                          csv_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')], axis=1)

    filter_feature = ['reporttime', 'hospital_expire_flag']
    features = []
    for x in csv_data.columns:
        if x not in filter_feature:
            features.append(x)
    train_data_x = csv_data[features]
    train_data_y = csv_data['hospital_expire_flag']
    train_data_y.to_csv('lb_mimic3_y.csv')
    features_mode = {}
    for f in features:
        features_mode[f] = list(train_data_x[f].dropna().mode().values)[0]
    train_data_x.fillna(features_mode, inplace=True)
    train_data_x.to_csv('lb_mimic3_x_fill.csv')

def save_data_mimic3():
    data = pd.read_csv("lb_mimic3_x_fill.csv",index_col=0)
    datay = pd.read_csv("lb_mimic3_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbpm', 'nbps', 'nbpd', 'abpm', 'abps',
       'abpd', 'fio2', 'temperaturef', 'cvp', 'peepset', 'meanairwaypressure',
       'tidalvolumeobserved', 'mvalarmhigh', 'mvalarmlow', 'apneainterval',
       'pawhigh', 'peakinsppressure', 'respiratoryratespontaneous',
       'minutevolume', 'vtihigh', 'respiratoryratetotal',
       'tidalvolumespontaneous', 'glucosefingerstick', 'it',
       'respiratoryrateset', 'hralarmlow', 'hralarmhigh', 'hematocrit',
       'potassium', 'sodium', 'creatinine', 'chloride', 'ureanitrogen',
       'bicarbonate', 'plateletcount', 'aniongap', 'whitebloodcells',
       'hemoglobin', 'glucose', 'mchc', 'redbloodcells', 'mch', 'mcv', 'rdw',
       'magnesium', 'calciumtotal', 'phosphate', 'ph', 'baseexcess',
       'calculatedtotalco2', 'po2', 'pco2', 'ptt', 'inr', 'pt',
       'bilirubintotal', 'freecalcium']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    x = []
    x1 = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    for i in range(data.index.size):
        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']
        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['tidalvolumeobserved'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['tidalvolumespontaneous'])
        x2.append(data1.iloc[i]['glucosefingerstick'])
        x2.append(data1.iloc[i]['it'])
        x2.append(data1.iloc[i]['respiratoryrateset'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['ph'])
        x2.append(data1.iloc[i]['baseexcess'])
        x2.append(data1.iloc[i]['calculatedtotalco2'])
        x2.append(data1.iloc[i]['po2'])
        x2.append(data1.iloc[i]['pco2'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['freecalcium'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        x1.append(x2)
        x2 = []
        if (i + 1 < data1.index.size and hadmid != data1.iloc[i + 1]['hadm_id']) or i == data1.index.size - 1:
            x.append(x1)
            y.append(datay.iloc[i]['hospital_expire_flag'])
            x1 = []

    x = list(map(lambda i: torch.tensor(i), x))
    lens = list(map(len, x))
    lens = np.array(lens)
    lens = torch.from_numpy(lens)
    padded_sequence = nn_utils.rnn.pad_sequence(x, batch_first=True)
    y_array = np.array(y)
    y_torch = torch.from_numpy(y_array)

def csv_to_fill_mimic4():
    csv_data = pd.read_csv("lb_mimic4.csv", low_memory=False)
    csv_data = pd.concat([csv_data[['reporttime', 'hadm_id','anchor_age','gender', 'hospital_expire_flag']],
                          csv_data.select_dtypes(include=['float']).apply(pd.to_numeric, downcast='float')], axis=1)

    filter_feature = ['reporttime', 'hospital_expire_flag']
    features = []
    for x in csv_data.columns:  # 取特征
        if x not in filter_feature:
            features.append(x)
    filter_feature = ['reporttime']
    train_data_x = csv_data[features]
    train_data_y = csv_data['hospital_expire_flag']
    train_data_y.to_csv('lb_mimic4_y.csv')
    features_mode = {}
    for f in features:
        features_mode[f] = list(train_data_x[f].dropna().mode().values)[0]
    train_data_x.fillna(features_mode, inplace=True)
    train_data_x.to_csv('lb_mimic4_x_fill.csv')

def save_data_mimic4():
    data = pd.read_csv("lb_mimic4_x_fill.csv",index_col=0)
    datay = pd.read_csv("lb_mimic4_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbps', 'nbpd', 'nbpm', 'abpm', 'abps',
       'abpd', 'temperaturef', 'fio2', 'peepset', 'tidalvolume',
       'minutevolume', 'meanairwaypressure', 'peakinsppressure', 'mvalarmlow',
       'mvalarmhigh', 'apneainterval', 'pawhigh', 'respiratoryratespontaneous',
       'vtihigh', 'respiratoryratetotal', 'fspnhigh', 'cvp', 'glucosefs',
       'flowrate', 'hralarmlow', 'hralarmhigh', 'spo2alarmlow', 'hematocrit',
       'creatinine', 'plateletcount', 'hemoglobin', 'whitebloodcells',
       'ureanitrogen', 'mchc', 'redbloodcells', 'mcv', 'mch', 'rdw',
       'potassium', 'sodium', 'chloride', 'bicarbonate', 'aniongap', 'glucose',
       'calciumtotal', 'magnesium', 'phosphate', 'inr', 'pt',
       'alanineaminotransferase', 'asparateaminotransferase', 'ptt',
       'bilirubintotal', 'neutrophils', 'lymphocytes', 'monocytes',
       'eosinophils']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    x = []
    x1 = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    for i in range(data.index.size):
        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']
        x2 = []#特征
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['tidalvolume'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['fspnhigh'])
        x2.append(data1.iloc[i]['glucosefs'])
        x2.append(data1.iloc[i]['flowrate'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['spo2alarmlow'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['alanineaminotransferase'])
        x2.append(data1.iloc[i]['asparateaminotransferase'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['neutrophils'])
        x2.append(data1.iloc[i]['lymphocytes'])
        x2.append(data1.iloc[i]['monocytes'])
        x2.append(data1.iloc[i]['eosinophils'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        x1.append(x2)
        x2 = []
        if (i + 1 < data1.index.size and hadmid != data1.iloc[i + 1]['hadm_id']) or i == data1.index.size - 1:  # 如果下一个就不是了,或者到头了，那么赶紧把这个装满了值的x1加到x里面
            x.append(x1)
            y.append(datay.iloc[i]['hospital_expire_flag'])
            x1 = []

    del data
    del data1
    del data2
    del datay

    y_array = np.array(y)
    y_torch = torch.from_numpy(y_array)
    pickle.dump(y_torch, open('lb_mimic4_y.p', 'wb'), protocol=4)
    del y_torch

    lens = list(map(len, x))
    lens = np.array(lens)
    lens = torch.from_numpy(lens)
    pickle.dump(lens, open('lb_mimic4_len.p', 'wb'), protocol=4)
    del lens

    x = list(map(lambda i: torch.tensor(i), x))
    padded_sequence = nn_utils.rnn.pad_sequence(x, batch_first=True)
    del x
    pickle.dump(padded_sequence, open('lb_mimic4_x.p', 'wb'), protocol=4)
    del padded_sequence


def save_data_mimic3_for_tra():
    data = pd.read_csv("lb_mimic3_x_fill.csv",index_col=0)
    datay = pd.read_csv("lb_mimic3_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbpm', 'nbps', 'nbpd', 'abpm', 'abps',
       'abpd', 'fio2', 'temperaturef', 'cvp', 'peepset', 'meanairwaypressure',
       'tidalvolumeobserved', 'mvalarmhigh', 'mvalarmlow', 'apneainterval',
       'pawhigh', 'peakinsppressure', 'respiratoryratespontaneous',
       'minutevolume', 'vtihigh', 'respiratoryratetotal',
       'tidalvolumespontaneous', 'glucosefingerstick', 'it',
       'respiratoryrateset', 'hralarmlow', 'hralarmhigh', 'hematocrit',
       'potassium', 'sodium', 'creatinine', 'chloride', 'ureanitrogen',
       'bicarbonate', 'plateletcount', 'aniongap', 'whitebloodcells',
       'hemoglobin', 'glucose', 'mchc', 'redbloodcells', 'mch', 'mcv', 'rdw',
       'magnesium', 'calciumtotal', 'phosphate', 'ph', 'baseexcess',
       'calculatedtotalco2', 'po2', 'pco2', 'ptt', 'inr', 'pt',
       'bilirubintotal', 'freecalcium']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    x = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    for i in range(data.index.size):
        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']
        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['tidalvolumeobserved'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['tidalvolumespontaneous'])
        x2.append(data1.iloc[i]['glucosefingerstick'])
        x2.append(data1.iloc[i]['it'])
        x2.append(data1.iloc[i]['respiratoryrateset'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['ph'])
        x2.append(data1.iloc[i]['baseexcess'])
        x2.append(data1.iloc[i]['calculatedtotalco2'])
        x2.append(data1.iloc[i]['po2'])
        x2.append(data1.iloc[i]['pco2'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['freecalcium'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        x.append(x2)
        y.append(datay.iloc[i]['hospital_expire_flag'])
        x2 = []

    y_array = np.array(y)
    y_torch = torch.from_numpy(y_array)
    pickle.dump(y_torch, open('lb_mimic3_y_for_tra.p', 'wb'))

    x = torch.Tensor(x)
    pickle.dump(x, open('lb_mimic3_x_for_tra.p', 'wb'))

def save_data_mimic4_for_tra():
    data = pd.read_csv("lb_mimic4_x_fill.csv",index_col=0)
    datay = pd.read_csv("lb_mimic4_y.csv",index_col=0)
    z_scaler = lambda x: (x - np.mean(x)) / np.std(x)
    data2 = data[['anchor_age',
       'heig', 'hr', 'rr', 'spo2', 'nbps', 'nbpd', 'nbpm', 'abpm', 'abps',
       'abpd', 'temperaturef', 'fio2', 'peepset', 'tidalvolume',
       'minutevolume', 'meanairwaypressure', 'peakinsppressure', 'mvalarmlow',
       'mvalarmhigh', 'apneainterval', 'pawhigh', 'respiratoryratespontaneous',
       'vtihigh', 'respiratoryratetotal', 'fspnhigh', 'cvp', 'glucosefs',
       'flowrate', 'hralarmlow', 'hralarmhigh', 'spo2alarmlow', 'hematocrit',
       'creatinine', 'plateletcount', 'hemoglobin', 'whitebloodcells',
       'ureanitrogen', 'mchc', 'redbloodcells', 'mcv', 'mch', 'rdw',
       'potassium', 'sodium', 'chloride', 'bicarbonate', 'aniongap', 'glucose',
       'calciumtotal', 'magnesium', 'phosphate', 'inr', 'pt',
       'alanineaminotransferase', 'asparateaminotransferase', 'ptt',
       'bilirubintotal', 'neutrophils', 'lymphocytes', 'monocytes',
       'eosinophils']].apply(z_scaler)
    data1 = pd.concat([data[['hadm_id','gender']],data2], axis=1)
    x = []
    y = []
    hadmid = data1.iloc[0]['hadm_id']
    for i in range(data.index.size):
        if i%1000 == 0:
            print(i)

        if hadmid != data1.iloc[i]['hadm_id']:
            hadmid = data1.iloc[i]['hadm_id']
        x2 = []
        x2.append(data1.iloc[i]['anchor_age'])
        x2.append(data1.iloc[i]['heig'])
        x2.append(data1.iloc[i]['hr'])
        x2.append(data1.iloc[i]['rr'])
        x2.append(data1.iloc[i]['spo2'])
        x2.append(data1.iloc[i]['nbpm'])
        x2.append(data1.iloc[i]['nbps'])
        x2.append(data1.iloc[i]['nbpd'])
        x2.append(data1.iloc[i]['abpm'])
        x2.append(data1.iloc[i]['abps'])
        x2.append(data1.iloc[i]['abpd'])
        x2.append(data1.iloc[i]['fio2'])
        x2.append(data1.iloc[i]['temperaturef'])
        x2.append(data1.iloc[i]['cvp'])
        x2.append(data1.iloc[i]['peepset'])
        x2.append(data1.iloc[i]['tidalvolume'])
        x2.append(data1.iloc[i]['minutevolume'])
        x2.append(data1.iloc[i]['meanairwaypressure'])
        x2.append(data1.iloc[i]['peakinsppressure'])
        x2.append(data1.iloc[i]['mvalarmlow'])
        x2.append(data1.iloc[i]['mvalarmhigh'])
        x2.append(data1.iloc[i]['apneainterval'])
        x2.append(data1.iloc[i]['pawhigh'])
        x2.append(data1.iloc[i]['respiratoryratespontaneous'])
        x2.append(data1.iloc[i]['vtihigh'])
        x2.append(data1.iloc[i]['respiratoryratetotal'])
        x2.append(data1.iloc[i]['fspnhigh'])
        x2.append(data1.iloc[i]['glucosefs'])
        x2.append(data1.iloc[i]['flowrate'])
        x2.append(data1.iloc[i]['hralarmlow'])
        x2.append(data1.iloc[i]['hralarmhigh'])
        x2.append(data1.iloc[i]['spo2alarmlow'])
        x2.append(data1.iloc[i]['hematocrit'])
        x2.append(data1.iloc[i]['creatinine'])
        x2.append(data1.iloc[i]['plateletcount'])
        x2.append(data1.iloc[i]['hemoglobin'])
        x2.append(data1.iloc[i]['whitebloodcells'])
        x2.append(data1.iloc[i]['ureanitrogen'])
        x2.append(data1.iloc[i]['mchc'])
        x2.append(data1.iloc[i]['redbloodcells'])
        x2.append(data1.iloc[i]['mcv'])
        x2.append(data1.iloc[i]['mch'])
        x2.append(data1.iloc[i]['rdw'])
        x2.append(data1.iloc[i]['potassium'])
        x2.append(data1.iloc[i]['sodium'])
        x2.append(data1.iloc[i]['chloride'])
        x2.append(data1.iloc[i]['bicarbonate'])
        x2.append(data1.iloc[i]['aniongap'])
        x2.append(data1.iloc[i]['glucose'])
        x2.append(data1.iloc[i]['calciumtotal'])
        x2.append(data1.iloc[i]['magnesium'])
        x2.append(data1.iloc[i]['phosphate'])
        x2.append(data1.iloc[i]['inr'])
        x2.append(data1.iloc[i]['pt'])
        x2.append(data1.iloc[i]['alanineaminotransferase'])
        x2.append(data1.iloc[i]['asparateaminotransferase'])
        x2.append(data1.iloc[i]['ptt'])
        x2.append(data1.iloc[i]['bilirubintotal'])
        x2.append(data1.iloc[i]['neutrophils'])
        x2.append(data1.iloc[i]['lymphocytes'])
        x2.append(data1.iloc[i]['monocytes'])
        x2.append(data1.iloc[i]['eosinophils'])
        if data1.iloc[i]['gender'] == 'f' or data1.iloc[i]['gender'] == 'F':
            x2.append(0)
        else:
            x2.append(1)
        x.append(x2)
        y.append(datay.iloc[i]['hospital_expire_flag'])
        x2 = []

    del data
    del data1
    del data2
    del datay

    y_array = np.array(y)
    y_torch = torch.from_numpy(y_array)
    pickle.dump(y_torch, open('lb_mimic4_y_for_tra.p', 'wb'), protocol=4)
    del y_torch
    del y_array
    del y

    x = torch.Tensor(x)
    pickle.dump(x, open('lb_mimic4_x_for_tra.p', 'wb'), protocol=4)
    del x

