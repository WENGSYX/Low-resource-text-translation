import os
import pandas as pd

with open(r'测试集\idzh\test.idzh.id.xml','r',encoding='utf-8') as f:
    id = f.read().split('\n')[4:-3]

data = []
with open('zh_id.txt','w',encoding='utf-8') as f:
    for i in range(10000):
        sentence = id[i].split('>')[1].split('<')[0].replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')
        data.append({'sen1':sentence,'sen2':''})
        f.write(sentence)
        f.write('\n')
data = pd.DataFrame(data)
data.to_csv('id_zh_test.csv')

with open(r'测试集\zhid\test.zhid.zh.xml', 'r', encoding='utf-8') as f:
    zh = f.read().split('\n')[4:-3]

data = []
with open('zh_id.txt','w',encoding='utf-8') as f:
    for i in range(10000):
        sentence = zh[i].split('>')[1].split('<')[0].replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')
        data.append({'sen1':sentence,'sen2':''})
        f.write(sentence)
        f.write('\n')
data = pd.DataFrame(data)
data.to_csv('zh_id_test.csv')


with open(r'测试集\mszh\test.mszh.ms.xml', 'r', encoding='utf-8') as f:
    ms = f.read().split('\n')[4:-3]

data = []
with open('ms_zh.txt','w',encoding='utf-8') as f:
    for i in range(10000):
        sentence = ms[i].split('>')[1].split('<')[0].replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')
        data.append({'sen1':sentence,'sen2':''})
        f.write(sentence)
        f.write('\n')
data = pd.DataFrame(data)
data.to_csv('ms_zh_test.csv')

with open(r'测试集\zhms\test.zhms.zh.xml', 'r', encoding='utf-8') as f:
    zh = f.read().split('\n')[4:-3]

data = []
with open('zh_ms.txt','w',encoding='utf-8') as f:
    for i in range(10000):
        sentence = zh[i].split('>')[1].split('<')[0].replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')
        data.append({'sen1':sentence,'sen2':''})
        f.write(sentence)
        f.write('\n')
data = pd.DataFrame(data)
data.to_csv('zh_ms_test.csv')
