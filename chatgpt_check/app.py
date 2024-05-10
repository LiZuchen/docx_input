# Copyright (c) Hello-SimpleAI Org. 2023.
# Licensed under the Apache License, Version 2.0.

import pickle
import re
from typing import Callable, List, Tuple

import gradio as gr
import matplotlib
from nltk.data import load as nltk_load
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


NLTK = nltk_load('data/english.pickle')
sent_cut_en = NLTK.tokenize
LR_GLTR_EN, LR_PPL_EN, LR_GLTR_ZH, LR_PPL_ZH = [
    pickle.load(open(f'data/{lang}-gpt2-{name}.pkl', 'rb'))
    for lang, name in [('en', 'gltr'), ('en', 'ppl'), ('zh', 'gltr'), ('zh', 'ppl')]
]

NAME_EN = 'gpt2'
TOKENIZER_EN = GPT2Tokenizer.from_pretrained(NAME_EN)
MODEL_EN = GPT2LMHeadModel.from_pretrained(NAME_EN)

NAME_ZH = 'D:\PyProject\docx_input\chatgpt_check\model\Wenzhong-GPT2-110M'
TOKENIZER_ZH = GPT2Tokenizer.from_pretrained(NAME_ZH)
MODEL_ZH = GPT2LMHeadModel.from_pretrained(NAME_ZH)
name='16021088_张文煊_基于实体链接和关系抽取的文本知识提取算法研究（最终版）'

# code borrowed from https://github.com/blmoistawinde/HarvestText
def sent_cut_zh(para: str) -> List[str]:
    para = re.sub('([。！？\?!])([^”’)\]）】])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{3,})([^”’)\]）】….])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…+)([^”’)\]）】….])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?!]|\.{3,}|\…+)([”’)\]）】])([^，。！？\?….])', r'\1\2\n\3', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    sentences = para.split("\n")
    sentences = [sent.strip() for sent in sentences]
    sentences = [sent for sent in sentences if len(sent.strip()) > 0]
    return sentences


CROSS_ENTROPY = torch.nn.CrossEntropyLoss(reduction='none')


def gpt2_features(
    text: str, tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, sent_cut: Callable
) -> Tuple[List[int], List[float]]:
    # Tokenize
    input_max_length = tokenizer.model_max_length - 2
    token_ids, offsets = list(), list()
    sentences = sent_cut(text)
    for s in sentences:
        tokens = tokenizer.tokenize(s)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        difference = len(token_ids) + len(ids) - input_max_length
        if difference > 0:
            ids = ids[:-difference]
        offsets.append((len(token_ids), len(token_ids) + len(ids)))  # 左开右闭
        token_ids.extend(ids)
        if difference >= 0:
            break

    input_ids = torch.tensor([tokenizer.bos_token_id] + token_ids)
    logits = model(input_ids).logits
    # Shift so that n-1 predict n
    shift_logits = logits[:-1].contiguous()
    shift_target = input_ids[1:].contiguous()
    loss = CROSS_ENTROPY(shift_logits, shift_target)

    all_probs = torch.softmax(shift_logits, dim=-1)
    sorted_ids = torch.argsort(all_probs, dim=-1, descending=True)  # stable=True
    expanded_tokens = shift_target.unsqueeze(-1).expand_as(sorted_ids)
    indices = torch.where(sorted_ids == expanded_tokens)
    rank = indices[-1]
    counter = [
        rank < 10,
        (rank >= 10) & (rank < 100),
        (rank >= 100) & (rank < 1000),
        rank >= 1000
    ]
    #词语的四个rank的值gtlr
    counter = [c.long().sum(-1).item() for c in counter]


    # compute different-level ppl
    text_ppl = loss.mean().exp().item()
    sent_ppl = list()
    for start, end in offsets:
        nll = loss[start: end].sum() / (end - start)
        sent_ppl.append(nll.exp().item())
    max_sent_ppl = max(sent_ppl)
    sent_ppl_avg = sum(sent_ppl) / len(sent_ppl)
    if len(sent_ppl) > 1:
        sent_ppl_std = torch.std(torch.tensor(sent_ppl)).item()
    else:
        sent_ppl_std = 0

    mask = torch.tensor([1] * loss.size(0))
    step_ppl = loss.cumsum(dim=-1).div(mask.cumsum(dim=-1)).exp()
    max_step_ppl = step_ppl.max(dim=-1)[0].item()
    step_ppl_avg = step_ppl.sum(dim=-1).div(loss.size(0)).item()
    if step_ppl.size(0) > 1:
        step_ppl_std = step_ppl.std().item()
    else:
        step_ppl_std = 0
    ppls = [
        text_ppl, max_sent_ppl, sent_ppl_avg, sent_ppl_std,
        max_step_ppl, step_ppl_avg, step_ppl_std
    ]
    #7个ppl的相关指标
    return counter, ppls  # type: ignore


def lr_predict(
    f_gltr: List[int], f_ppl: List[float], lr_gltr: LogisticRegression, lr_ppl: LogisticRegression,
    id_to_label: List[str]
) -> List:
    x_gltr = np.asarray([f_gltr])
    gltr_label = lr_gltr.predict(x_gltr)[0]
    gltr_prob = lr_gltr.predict_proba(x_gltr)[0, gltr_label]
    x_ppl = np.asarray([f_ppl])
    ppl_label = lr_ppl.predict(x_ppl)[0]
    ppl_prob = lr_ppl.predict_proba(x_ppl)[0, ppl_label]
    return [id_to_label[gltr_label], gltr_prob, id_to_label[ppl_label], ppl_prob]


def predict_en(text: str) -> List:
    with torch.no_grad():
        feat = gpt2_features(text, TOKENIZER_EN, MODEL_EN, sent_cut_en)
    out = lr_predict(*feat, LR_GLTR_EN, LR_PPL_EN, ['Human', 'ChatGPT'])
    return out


def predict_zh(text: str) -> List:
    with torch.no_grad():
        feat = gpt2_features(text, TOKENIZER_ZH, MODEL_ZH, sent_cut_zh)
    out = lr_predict(*feat, LR_GLTR_ZH, LR_PPL_ZH, ['人类', 'ChatGPT'])
    return out


with gr.Blocks() as demo:
    gr.Markdown(
        """
        ## ChatGPT Detector 🔬 (Linguistic version / 语言学版)

        Visit our project on Github: [chatgpt-comparison-detection project](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)<br>
        欢迎在 Github 上关注我们的 [ChatGPT 对比与检测项目](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)<br>
        We provide three kinds of detectors, all in Bilingual / 我们提供了三个版本的检测器，且都支持中英文:
        - [QA version / 问答版](https://www.modelscope.cn/studios/simpleai/chatgpt-detector-qa)<br>
            detect whether an **answer** is generated by ChatGPT for certain **question**, using PLM-based classifiers / 判断某个**问题的回答**是否由ChatGPT生成，使用基于PTM的分类器来开发;
        - [Sinlge-text version / 独立文本版](https://www.modelscope.cn/studios/simpleai/chatgpt-detector-single)<br>
            detect whether a piece of text is ChatGPT generated, using PLM-based classifiers / 判断**单条文本**是否由ChatGPT生成，使用基于PTM的分类器来开发;
        - [**Linguistic version / 语言学版** (👈 Current / 当前使用)](https://www.modelscope.cn/studios/simpleai/chatgpt-detector-ling)<br>
            detect whether a piece of text is ChatGPT generated, using linguistic features / 判断**单条文本**是否由ChatGPT生成，使用基于语言学特征的模型来开发;

        """
    )

    with gr.Tab("中文版"):
        gr.Markdown(
            """
            ## 介绍:
            两个逻辑回归模型, 分别使用以下两种特征:
            1. [GLTR](https://aclanthology.org/P19-3019) Test-2, 每个词的语言模型预测排名分桶, top 10, 10-100, 100-1000, 1000+.
            2. 基于语言模型困惑度 (PPL), 整个文本的PPL、单个句子的PPL等特征.

            中文语言模型使用 闻仲 [Wenzhong-GPT2-110M](https://huggingface.co/IDEA-CCNL/Wenzhong-GPT2-110M).

            注意: 在`文本`栏中输入更多的文本，可以让预测更准确哦！
            """
        )
        with open('D:\PyProject\docx_input\\file\\txt\\'+name+'.txt','r',encoding='utf-8') as f:
            content=f.read()
        f.close()
        a2 = gr.Textbox(
            lines=5, label='文本',
            value=content
        )
        button2 = gr.Button("🤖 预测!")
        gr.Markdown("GLTR (中文测试集准确率 86.39%)")
        label2_gltr = gr.Textbox(lines=1, label='预测结果 🎃')
        score2_gltr = gr.Textbox(lines=1, label='模型概率')

        gr.Markdown("PPL (中文测试集准确率 59.04%, 持续优化中...)")
        label2_ppl = gr.Textbox(lines=1, label='PPL 预测结果 🎃')
        score2_ppl = gr.Textbox(lines=1, label='PPL 模型概率')


    with gr.Tab("English"):
        gr.Markdown(
            """
            ## Introduction:
            Two Logistic regression models trained with two kinds of features:
            1. [GLTR](https://aclanthology.org/P19-3019) Test-2, Language model predict token rank top-k buckets, top 10, 10-100, 100-1000, 1000+.
            2. PPL-based, text ppl, sentence ppl, etc.

            English LM is [GPT2-small](https://huggingface.co/gpt2).

            Note: Providing more text to the `Text` box can make the prediction more accurate!
            """
        )
        a1 = gr.Textbox(
            lines=5, label='Text',
            value="There are a few things that can help protect your credit card information from being misused when you give it to a restaurant or any other business:\n\nEncryption: Many businesses use encryption to protect your credit card information when it is being transmitted or stored. This means that the information is transformed into a code that is difficult for anyone to read without the right key."
        )
        button1 = gr.Button("🤖 Predict!")
        gr.Markdown("GLTR")
        label1_gltr = gr.Textbox(lines=1, label='GLTR Predicted Label 🎃')
        score1_gltr = gr.Textbox(lines=1, label='GLTR Probability')
        gr.Markdown("PPL")
        label1_ppl = gr.Textbox(lines=1, label='PPL Predicted Label 🎃')
        score1_ppl = gr.Textbox(lines=1, label='PPL Probability')

    button1.click(predict_en, inputs=[a1], outputs=[label1_gltr, score1_gltr, label1_ppl, score1_ppl])
    button2.click(predict_zh, inputs=[a2], outputs=[label2_gltr, score2_gltr, label2_ppl, score2_ppl])



    # Page Count
    gr.Markdown("""
                <center><a href="https://clustrmaps.com/site/1bsdd" title="Visit tracker"><img src="//clustrmaps.com/map_v2.png?cl=080808&w=a&t=tt&d=NvxUHBTxY0ECXEuebgz8Ym8ynpVtduq59ENXoQpFh74&co=ffffff&ct=808080"/></a></center>
                """)

matplotlib.use('TkAgg')
demo.launch()


# with open('D:\PyProject\docx_input\data_cache\\aicheck\\' + name + '_ai.txt', 'w') as f:
#     print([label2_gltr, score2_gltr, label2_ppl, score2_ppl], file=f)
# f.close()
# To create a public link, set `share=True` in `launch()`.