import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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

import time
from sklearn.model_selection import *
from transformers import *
from torch.autograd import Variable
import sacrebleu

CFG = { #训练的参数配置
    'fold_num': 5, #五折交叉验证
    'seed': 2,
    'model': 'm2m', #预训练模型
    'max_len': 80, #文本截断的最大长度
    'epochs': 8,
    'train_bs': 1, #batch_size，可根据自己的显存调整
    'valid_bs': 1,
    'lr': 1e-5, #学习率
    'num_workers': 0,
    'accum_iter': 8, #梯度累积，相当于将batch_size*2
    'weight_decay': 1e-4, #权重衰减，防止过拟合
    'device': 0,
    'sen1':'id',
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

train_df =  pd.read_csv('{}_{}.csv'.format(CFG['sen1'],CFG['sen2']))
valid_df =  pd.read_csv('{}_{}_dev.csv'.format(CFG['sen1'],CFG['sen2']))
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

        sen2 = self.df.sen2.values[idx]
        sen1 = self.df.sen1.values[idx]

        return sen1,sen2

def collate_fn(data):
    input_ids, attention_mask, token_type_ids,label = [], [], [],[]
    for x in data:
        text = sen1_tokenizer(x[0],padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(text['input_ids'].squeeze().tolist())
        attention_mask.append(text['attention_mask'].squeeze().tolist())
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    for x in data:
        l = sen2_tokenizer(x[-1],padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        label.append(l.squeeze().tolist())
    label = torch.tensor(label)
    return input_ids, attention_mask,label,data

class AverageMeter:  # 为了tqdm实时显示loss和acc
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(model, fgm,pgd,train_loader):  # 训练一个epoch
    model.train()

    losses = AverageMeter()

    optimizer.zero_grad()

    tk = tqdm(train_loader, total=len(train_loader), position=0, leave=True)

    for step, (input_ids, attention_mask,y,data) in enumerate(tk):
        input_ids, attention_mask, y = input_ids.to(device), attention_mask.to(
            device),y.to(device).long()

        output = model(input_ids, attention_mask,labels=y)
        loss = criterion(output.logits.view(-1, model.config.vocab_size), y.view(-1))
        loss.backward()
        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):  # 梯度累加
            optimizer.step()
            optimizer.zero_grad()



        losses.update(loss.item() * CFG['accum_iter'], y.size(0))

        tk.set_postfix(loss=losses.avg)

    return losses.avg


def test_model(model, val_loader):  # 验证
    model.eval()

    bleus = AverageMeter()
    accs = AverageMeter()
    y_truth, y_pred = [], []

    with torch.no_grad():
        tk = tqdm(val_loader, total=len(val_loader), position=0, leave=True)
        for idx, (input_ids, attention_mask, y,data) in enumerate(tk):

            encoded = sen1_tokenizer(data[0][0], return_tensors="pt").to(device)
            output_tokens = model.generate(**encoded,forced_bos_token_id=sen2_tokenizer.get_lang_id(CFG['sen2']))

            output = sen2_tokenizer.batch_decode(output_tokens,skip_special_tokens=True)[0]
            label = data[0][1]

            if CFG['sen2'] == 'ru' or CFG['sen2'] == 'bg':
                bleu = sacrebleu.corpus_bleu([' '.join(output.split(' '))], [' '.join(label.split(' '))])
            else:
                bleu = sacrebleu.corpus_bleu([' '.join(x for x in output)],[' '.join(x for x in label)])
            if idx % 10 == 0:
                print(output)
                print(label)
                print(bleu.format())
            bleus.update(bleu.precisions[3])
            tk.set_postfix(bleu=bleus.avg)
    return bleus.avg

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
            if param.requires_grad and emb_name2 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
            if param.requires_grad and emb_name3 in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.backup
                param.data = self.backup[name]
            if param.requires_grad and emb_name2 in name:
                assert name in self.backup
                param.data = self.backup[name]
            if param.requires_grad and emb_name3 in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
            if param.requires_grad and emb_name2 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)
            if param.requires_grad and emb_name3 in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name1='bert.embeddings.word_embeddings.weight',emb_name2='bert.embeddings.position_embeddings.weight',emb_name3='bert.embeddings.token_type_embeddings.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name1 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
            if param.requires_grad and emb_name2 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
            if param.requires_grad and emb_name3 in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            targets = torch.eye(2)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            targets = torch.eye(2)[targets.reshape(-1)].to(device)
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class In_trust_Loss(nn.Module):
    def __init__(self, alpha=1, beta=0.8,delta=0.5, num_classes=35):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.delta = delta
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        #self.crf = CRF(num_tags= num_classes, batch_first=True)
    def forward(self, logits,labels):

        #loss_mask = labels.gt(0)
        #Loss CRF
        ce = self.cross_entropy(logits,labels)
        #Loss In_trust
        active_logits = logits.view(-1,self.num_classes)
        active_labels = labels.view(-1)

        pred = F.softmax(active_logits, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(active_labels,self.num_classes).float()
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        dce = (-1*torch.sum(pred * torch.log(pred*self.delta + label_one_hot*(1-self.delta)), dim=1))

        # Loss

        loss = self.alpha * ce - self.beta * dce.mean()
        return loss

train = train_df.loc[:]
val = valid_df.loc[:]

train_set = MyDataset(train)
val_set = MyDataset(val)

model = M2M100ForConditionalGeneration.from_pretrained(CFG['model']).to(device)  # 模型

#model.load_state_dict(torch.load('id_zhm2m_1_2.2833103212344823.pt'))

scaler = GradScaler()
optimizer = Adafactor(model.parameters(), weight_decay=CFG['weight_decay'])  # AdamW优化器
criterion = In_trust_Loss(num_classes=model.config.vocab_size)
val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                            num_workers=CFG['num_workers'])
train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])


fgm = FGM(model)
pgd = PGD(model)

for epoch in range(CFG['epochs']):
    val_loader = DataLoader(val_set, batch_size=CFG['valid_bs'], collate_fn=collate_fn, shuffle=False,
                            num_workers=CFG['num_workers'])
    train_loader = DataLoader(train_set, batch_size=CFG['train_bs'], collate_fn=collate_fn, shuffle=True,
                              num_workers=CFG['num_workers'])

    print('epoch:', epoch)
    time.sleep(0.2)
    train_loss= train_model(model, fgm, pgd, train_loader)
    val_loss = test_model(model, val_loader)
    torch.save(model.state_dict(), '{}_{}m2m_{}_{}.pt'.format(CFG['sen1'], CFG['sen2'], str(epoch), str(val_loss)))

