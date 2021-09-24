import pandas as pd
import os
from tqdm import tqdm
data = []
with open('训练集/train.id-zh','r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        data.append({'sen1':line.split('\t')[0],'sen2':line.split('\t')[1]})

data = pd.DataFrame(data)
data.to_csv('id_zh.csv')
data = []
with open('训练集/train.id-zh','r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        data.append({'sen1':line.split('\t')[1],'sen2':line.split('\t')[0]})

data = pd.DataFrame(data)
data.to_csv('zh_id.csv')

data = []
with open('训练集/train.ms-zh','r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        data.append({'sen2':line.split('\t')[0],'sen1':line.split('\t')[1]})

data = pd.DataFrame(data)
data.to_csv('ms_zh.csv')
data = []
with open('训练集/train.ms-zh','r',encoding='utf-8') as f:
    for line in tqdm(f.readlines()):
        data.append({'sen2':line.split('\t')[1],'sen1':line.split('\t')[0]})

data = pd.DataFrame(data)
data.to_csv('zh_ms.csv')