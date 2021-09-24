import pandas as pd
import os



with open('开发集/idzh/val.id-zh.id.xml','r',encoding='utf-8') as f:
    id = f.read().split('\n')[4:-3]
with open('开发集/idzh/val.id-zh.zh.xml','r',encoding='utf-8') as f:
    zh = f.read().split('\n')[4:-3]
data = []
for i in range(1000):
    data.append({'sen1':id[i].split('>')[1].split('<')[0],'sen2':zh[i].split('>')[1].split('<')[0]})
data = pd.DataFrame(data)
data.to_csv('id_zh_dev.csv')

with open('开发集/zhid/val.zh-id.id.xml','r',encoding='utf-8') as f:
    id = f.read().split('\n')[4:-3]
with open('开发集/zhid/val.zh-id.zh.xml','r',encoding='utf-8') as f:
    zh = f.read().split('\n')[4:-3]
data = []
for i in range(1000):
    data.append({'sen2':id[i].split('>')[1].split('<')[0],'sen1':zh[i].split('>')[1].split('<')[0]})
data = pd.DataFrame(data)
data.to_csv('zh_id_dev.csv')



with open('开发集/mszh/val.ms-zh.ms.xml','r',encoding='utf-8') as f:
    ms = f.read().split('\n')[4:-3]
with open('开发集/mszh/val.ms-zh.zh.xml','r',encoding='utf-8') as f:
    zh = f.read().split('\n')[4:-3]
data = []
for i in range(1000):
    data.append({'sen1':ms[i].split('>')[1].split('<')[0],'sen2':zh[i].split('>')[1].split('<')[0]})
data = pd.DataFrame(data)
data.to_csv('ms_zh_dev.csv')

with open('开发集/zhms/val.zh-ms.ms.xml','r',encoding='utf-8') as f:
    ms = f.read().split('\n')[4:-3]
with open('开发集/zhms/val.zh-ms.zh.xml','r',encoding='utf-8') as f:
    zh = f.read().split('\n')[4:-3]
data = []
for i in range(1000):
    data.append({'sen2':ms[i].split('>')[1].split('<')[0],'sen1':zh[i].split('>')[1].split('<')[0]})
data = pd.DataFrame(data)
data.to_csv('zh_ms_dev.csv')