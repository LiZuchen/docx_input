# -*- coding: utf-8 -*-
import numpy as np
import torch
import torchvision
import torchaudio
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



# embedding()
# cal_ppl_bygpt2()