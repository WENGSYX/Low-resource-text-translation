

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import random
import os
import time
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable
import sacrebleu
os.chdir('D:\python\比赛\天池\翻译')
CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 2,
    'model': 'm2m', #预训练模型
    'max_len': 300, #文本截断的最大长度
    'epochs': 8,
    'train_bs': 20, #batch_size，可根据自己的显存调整
    'valid_bs': 1,
    'lr': 1e-4, #学习率
    'num_workers': 0,
    'accum_iter': 4, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
    'sen1':'ms',
    'sen2':'zh'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG['seed']) #固定随机种子

torch.cuda.set_device(CFG['device'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

valid_df =  pd.read_csv('{}_{}_test.csv'.format(CFG['sen1'],CFG['sen2']))
sen1_tokenizer = M2M100Tokenizer.from_pretrained(CFG['model'])
sen1_tokenizer.src_lang = CFG['sen1']
sen2_tokenizer = M2M100Tokenizer.from_pretrained(CFG['model'])
sen2_tokenizer.src_lang = CFG['sen2']
class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        sen1 = self.df.sen1.values[idx].replace('&amp;','&').replace('&lt;','<').replace('&gt;','>')

        return sen1

def collate_fn(data):
    return data


val = valid_df.loc[:]

val_set = MyDataset(val)

model = M2M100ForConditionalGeneration.from_pretrained(CFG['sen1']+'_'+CFG['sen2']).to(device)  # 模型
#model.load_state_dict(torch.load('id_zhm2m_1_2.2833103212344823.pt'))

val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                        num_workers=CFG['num_workers'])

pred = []
with torch.no_grad():
    tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
    for idx, (data) in enumerate(tk):
        encoded = sen1_tokenizer(data, return_tensors="pt").to(device)
        output_tokens = model.generate(**encoded, forced_bos_token_id=sen2_tokenizer.get_lang_id(CFG['sen2']))

        output = sen2_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        print(output)
        pred.append(output)


import opencc
converter = opencc.OpenCC('t2s.json')
with open('{}{}.xml'.format(CFG['sen1'],CFG['sen2']),'w',encoding='utf-8') as f:
    f.write("<?xml version='1.0' encoding='utf-8'?>")
    f.write('\n')
    f.write('<refset setid="{}_{}_test" srclang="{}" trglang="{}">'.format(CFG['sen1'],CFG['sen2'],CFG['sen1'],CFG['sen2']))
    f.write('\n')
    f.write('<system site="个人" sysid="翻译系统1.0">')
    f.write('\n')
    f.write('系统:WINDOWS11 CPU:AMD 5800X *1 4.4HZ GPU:3090 内存:48G 运行时间 2小时27分24秒 技术概要:基于预训练模型微调 外部技术说明:使用python程序，基于transformer库和pytorch，对M2M模型微调')
    f.write('\n')
    f.write('</system>')
    f.write('\n')
    f.write('<DOC docid="testset" site="1" sysid="ref">')
    f.write('\n')
    f.write('<p>')
    f.write('\n')
    for i in range(0,10000):
        if pred[i]=='':
            f.write('<seg id="{}">{}</seg>'.format(i + 1,'hello'))
        else:
            f.write('<seg id="{}">{}</seg>'.format(i+1,converter.convert(pred[i].replace('&','&amp;').replace('<','&lt;').replace('>','&gt;'))))
        f.write('\n')
    f.write('</p>')
    f.write('\n')
    f.write('</DOC>')
    f.write('\n')
    f.write('</refset>')
