# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchaudio
from config.config import sents_sim_cache_on
from torch import nn
from transformers import BertTokenizer, BertForMaskedLM, GPT2LMHeadModel
import transformers
import datasets
from datasets import load_dataset
#----------------------------------------------------
# print(torch.__version__,
#       torchvision.__version__,
#       torchaudio.__version__,
#       datasets.__version__,
#       transformers.__version__)
# 2.2.1+cpu 0.17.1+cpu 2.2.1+cpu 2.18.0 4.38.2
from transformers import BertTokenizer
from transformers import BertModel
#加载预训练字典和分词方法
def bert_encode(a):
      tokenizer = BertTokenizer.from_pretrained(
          pretrained_model_name_or_path='bert-base-chinese',
          cache_dir=None,
          force_download=False,
      )
      # %%
      # 编码两个句子
      # out = tokenizer.encode(
      #       text=sents[0],
      #       text_pair=sents[1],
      #
      #       # 当句子长度大于max_length时,截断
      #       truncation=True,
      #
      #       # 一律补pad到max_length长度
      #       padding='max_length',
      #       add_special_tokens=True,
      #       max_length=30,
      #       return_tensors=None,
      # )
      # 增强的编码函数
      out = tokenizer.encode_plus(
            text=a[0],
            text_pair=a.paragraphs[0][0],#可以没有这一行

            # 当句子长度大于max_length时,截断
            truncation=True,

            # 一律补零到max_length长度
            padding='max_length',
            max_length=30,
            add_special_tokens=True,

            # 可取值tf,pt,np,默认为返回list
            return_tensors=None,

            # 返回token_type_ids
            return_token_type_ids=True,

            # 返回attention_mask
            return_attention_mask=True,

            # 返回special_tokens_mask 特殊符号标识
            return_special_tokens_mask=True,

            # 返回offset_mapping 标识每个词的起止位置,这个参数只能BertTokenizerFast使用
            # return_offsets_mapping=True,

            # 返回length 标识长度
            return_length=True,
      )

      # input_ids 就是编码后的词
      # token_type_ids 第一个句子和特殊符号的位置是0,第二个句子的位置是1
      # special_tokens_mask 特殊符号的位置是1,其他位置是0
      # attention_mask pad的位置是0,其他位置是1
      # length 返回句子长度
      for k, v in out.items():
            print(k, ':', v)

      attenmask=out['attention_mask']
      tokenizer.decode(out['input_ids'])
      print(out)
      print(attenmask)

      tokenizer.decode(out)#解码


class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset(path=r'D:\PyProject\docx_input\blockcsv_out_files', split=split)
        # from hf hub
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text']
        label = self.dataset[i]['label']

        return text, label
def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [int(i[1].split('_')[2]) for i in data]
    print("in collate_fn() :",len(sents),sents)
    print("in collate_fn() :",len(labels),labels)
    #编码
    token = BertTokenizer.from_pretrained('bert-base-chinese')
    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=500,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:编码之后的数字
    #attention_mask:是补零的位置是0,其他位置是1
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels
def embedding():
      dataset = Dataset('train')
      print("dataset长度：",len(dataset), "dateset[0]为",dataset[0])#('绪论', '绪论_Heading 1_126')
      token = BertTokenizer.from_pretrained('bert-base-chinese')
      # print(token)
      loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=16,
                                           collate_fn=collate_fn,
                                           shuffle=False,
                                           drop_last=True)

      for i, (input_ids, attention_mask, token_type_ids,labels) in enumerate(loader):
            break
      print(len(loader))
      print("input_ids.shape: ", input_ids.shape)
      print(input_ids)
      print(input_ids.shape, attention_mask.shape, token_type_ids.shape, labels)
      #加载预训练模型
      pretrained = BertModel.from_pretrained('bert-base-chinese')

      # 不训练,不需要计算梯度
      for param in pretrained.parameters():
            param.requires_grad_(False)

      # 模型试算
      out = pretrained(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids)

      '''
      一组16个句子，每个句子500个向量，一个向量768维
      '''
      #输出向量表示
      print(out.last_hidden_state.data)
      print(out.last_hidden_state.shape)
def cal_ppl():
    model = BertForMaskedLM.from_pretrained('bert-base-chinese')
    model.eval()
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    sentence = "我不会忘记和你一起奋斗的时光。"
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    sen_len = len(tokenize_input)
    sentence_loss = 0.
    for i, word in enumerate(tokenize_input):
        # add mask to i-th character of the sentence
        tokenize_input[i] = '[MASK]'
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

        output = model(mask_input)

        prediction_scores = output[0]
        softmax = nn.Softmax(dim=0)
        ps = softmax(prediction_scores[0, i]).log()
        word_loss = ps[tensor_input[0, i]]
        sentence_loss += word_loss.item()

        tokenize_input[i] = word
    ppl = np.exp(-sentence_loss / sen_len)
    print(ppl)

from torch.nn import CrossEntropyLoss
def cal_ppl_bygpt2(sents,name):
    # sens = ["今天是个好日子。", "天今子日。个是好", "这个婴儿有900000克呢。", "我不会忘记和你一起奋斗的时光。",
    #         "我不会记忘和你一起奋斗的时光。", "会我记忘和你斗起一奋的时光。","今天是个好日子,今天是个好日子,今天是个好日子,今天是个好日子"]

    sens=sents
    tokenizer = BertTokenizer.from_pretrained("D:\PyProject\docx_input\pretainedmodel\gpt2-chinese")
    model = GPT2LMHeadModel.from_pretrained("D:\PyProject\docx_input\pretainedmodel\gpt2-chinese")
    inputs = tokenizer(sens, padding='max_length', max_length=50, truncation=True, return_tensors="pt")
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).numpy().tolist()
    pplpath=r'D:\PyProject\docx_input\data_cache\ppllist_cache\\'
    with open(pplpath+name+'.txt', 'w') as f:
        print(ppl,file=f)
    return ppl


from transformers import AutoTokenizer, AutoModel
import numpy as np
# 加载BERT模型和分词器
# 定义计算相似度的函数
def calc_similarity(s1, s2):
    # 对句子进行分词，并添加特殊标记
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    inputs = tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True)

    # 将输入传递给BERT模型，并获取输出
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

    # 计算余弦相似度，并返回结果
    sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    # print("相似度",sim)
    return sim
# 测试函数
# similarity = calc_similarity(s1, s2)
def cal_sim_bybert(sentswithlabel,name):
    print(1)
    sentssim=[]
    for key in sentswithlabel.keys():
        for i in range(0,len(sentswithlabel[key])):
            if  i<len(sentswithlabel[key])-1 :
            # for j in range(i+1,len(sentswithlabel[key])):
                j=i+1
                sentssim.append([sentswithlabel[key][i],sentswithlabel[key][j],calc_similarity(sentswithlabel[key][i],sentswithlabel[key][j])])
                    # print(sentssim[-1],sentswithlabel[key][i],sentswithlabel[key][j])
    # print(sentssim[-1], sentswithlabel[key][i], sentswithlabel[key][j])
    # sentssimpath = r'D:\PyProject\docx_input\data_cache\sents_sim_cache\\'
    # with open(sentssimpath + name + '.txt', 'w') as f:
    #     print(sentssim, file=f)
    print(2)
    sents1=[]
    sents2=[]
    sim=[]
    for i in sentssim:
        sents1.append(i[0])
        sents2.append(i[1])
        sim.append(i[2])
    dataframe = pd.DataFrame({'sents1': sents1,'sents2': sents2,'sim':sim})
    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(r'D:\PyProject\docx_input\data_cache\sents_sim_cache\\' +
                     name + "sents_sim.csv", index=False, sep=',')

    return sentssim


# print(f"相似度：{similarity:.4f}")
# embedding()
# cal_ppl_bygpt2()
def cal_ppl_bygpt2_4gpt_gen(sents,name):
    sens = sents
    tokenizer = BertTokenizer.from_pretrained("D:\PyProject\docx_input\pretainedmodel\gpt2-chinese")
    model = GPT2LMHeadModel.from_pretrained("D:\PyProject\docx_input\pretainedmodel\gpt2-chinese")
    inputs = tokenizer(sens, padding='max_length', max_length=50, truncation=True, return_tensors="pt")
    bs, sl = inputs['input_ids'].size()
    outputs = model(**inputs, labels=inputs['input_ids'])
    logits = outputs[1]
    # Shift so that tokens < n predict n
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = inputs['input_ids'][:, 1:].contiguous()
    shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
    meanloss = loss.sum(1) / shift_attentions.sum(1)
    ppl = torch.exp(meanloss).numpy().tolist()
    pplpath = r'D:\PyProject\docx_input\chatgpt_gen_ppl\\'
    with open(pplpath + name + '.txt', 'w') as f:
        print(ppl, file=f)
    return ppl
def readgpt_gen():
    files=os.listdir('D:\PyProject\docx_input\chatgpt_gen')
    print(files)
    name='研究背景'
    file_path='D:\PyProject\docx_input\chatgpt_gen\\'+ name+'.txt'
    with open (file_path,'r',encoding='utf-8') as f:
        txt = []
        for line in f:
            txt.append(line.strip())
        print(txt)
    sents=[]
    for i in txt:
        linei=list(i.split('。'))
        for j in linei:
            if j!='':
                sents.append(j+'。')
    print(len(sents))
    cal_ppl_bygpt2_4gpt_gen(sents,name)
readgpt_gen()