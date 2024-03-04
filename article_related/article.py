# -*- coding: utf-8 -*-
import re

import config.config
import draw.task1_draw
import matplotlib
import pandas as pd
import seaborn
import sns as sns
from bert_embedding.bert_emb import cal_ppl_bygpt2, cal_sim_bybert
from draw.task1_draw import pplnum_drawbar
from sympy import stats
import matplotlib.pyplot as plt

class article:
    def __int__(self, name, paragraphs):
        self.name=name
        self.paragraphs=paragraphs

    def setname(self,name):
        self.name=name
    def getname(self):
        return self.name

    def setrws(self,rws):
        self.rws=rws
    def getrws(self):
        return self.rws

    def setkeyword(self,keyword):
        self.keyword=keyword
    def getkeyword(self):
        return self.keyword

    def setzy(self,zy):
        self.zy=zy
    def getzy(self):
        return self.zy

    def setmenu(self,menu):
        self.menu=menu
    def getmenu(self):
        return self.menu
    def menucheck(self):
        for m in self.menu:
            if len(m[0])>40:
                print("in menucheck 存在>40的标题",m[0])
    def setparagraphs(self,paragraphs):
        self.paragraphs=paragraphs
    def getparagraphs(self):
        return self.paragraphs

    def getblocks(self):
        return self.blocks
    def setblocks(self,blocks):
        self.blocks=blocks
    def blockstocsv(self):
        if config.config.blockstocsv_cache_on==0:
            label=[]
            text=[]
            for i in self.blocks:
                labeli="_".join(i[0:2])+'_'+str(i[2])
                for j in i[3]:
                    if j!='':
                        text.append(j)
                        label.append(labeli)
            dataframe = pd.DataFrame({'label': label, 'text': text})
            # 将DataFrame存储为csv,index表示是否显示行名，default=True
            dataframe.to_csv(r'D:\PyProject\docx_input\blockcsv_out_files\\'+
                             self.name+"_blocks.csv", index=False, sep=',')

    def setppllist(self, ppllist):
        self.ppllist = ppllist

    def pplcheck(self):
        draw_on=config.config.ppl_distribution_draw
        ppl_len_check_on=config.config.ppl_len_check_on
        #如果on，那么要设置一次ppl
        if ppl_len_check_on:
            if len(self.ppllist) == len(self.sents) :
                print("ppl check text ppllist len equal")
            else:
                print("ppl check text ppllist len not equal!")

        #ppl 分布计算和画图
        y=[0,0,0,0,0,0,0,0]
        for i in self.ppllist:
            if i>=0 and i<10:
            # 1
                y[0]+=1
            elif i>=10 and i<25:
            # 2
                y[1] += 1
            elif i >= 25 and i < 50:
            # 3
                y[2] += 1
            elif i >= 50 and i < 100:
            # 4
                y[3] += 1
            elif i >= 100 and i < 500:
            # 5
                y[4] += 1
                if config.config.ppl_large_show ==1:
                    print('100-500', ' in ',self.name," ",self.sents[sum(y) - 1])
            elif i>=500 and i<1000:
            # 6
                y[5] += 1
                print('500-1000',self.sents[sum(y)-1])
            elif i>=1000 and i<2000:
            # 7
                y[6] += 1
            else:
            # 8
                y[7] += 1
            #     print(self.text[self.ppllist.index(i)].encode('gbk', 'ignore').decode('gbk'))
        # print(y)
        self.ppl_distribution=y
        if draw_on:
            pplnum_drawbar(y,self.name+' ppl distribution')

    def totext(self):
        text = []
        for i in self.blocks:
            # labeli="_".join(i[0:2])+'_'+str(i[2])
            for j in i[3]:
                if j != '':
                    text.append(j)
        self.text=text
    def calppl(self):
        self.totext()
        #以后可能在计算没用
        if config.config.ppl_cache_on:
            filepath = r'D:\PyProject\docx_input\data_cache\ppllist_cache\\' + self.name + '.txt'
            with open(filepath, "r", encoding='gbk') as f:  # 打开文件
                data = f.read()  # 读取文件
                # print(data)
            # strlist=list(filter(None,re.split(' |,|\[|]|\n',data)))
            # floatlist=list(map(float,strlist))
            floatlist = eval(data)
            self.setppllist(floatlist)
        else:
            self.setppllist(cal_ppl_bygpt2(self.sents,self.name))

        self.pplcheck()

    def cal_sents_sim(self):

        if config.config.sents_sim_cache_on:
            filepath = r'D:\PyProject\docx_input\data_cache\sents_sim_cache\\' + self.name + '.txt'
            with open(filepath, "r", encoding='gbk') as f:  # 打开文件
                data = f.read()  # 读取文件
                    # print(data)
                sents_sim=eval(data)
                # strlist = list(filter(None, re.split('\', \'|\[|]', data)))
            # print(strlist)

            self.setsents_sim(sents_sim)

        else:
            self.setsents_sim(cal_sim_bybert(self.sentswithlabel,self.name))

        self.sents_sim_check()
    def sents_sim_check(self):
        for i in self.sents_sim:
            # print(i)
            if i[2]<0.6:
                print(i)
            if i[2]>0.95:
                print(i)
        return
    def tosents(self):
        sents_to_csv=config.config.sents_to_csv
        sents=[]
        sentswithlabel=dict()
        #block(内涵多个paragraps)
        #paragraph内涵多个sentences
        for i in self.blocks:
            # labeli = "_".join(i[0:2]) + '_' + str(i[2])
            # print(i[0:3])
            for j in i[3]:
                if j != ''and j[len(j)-1]=='。':
                    sentsi=list(j.split('。'))
                    for k in sentsi:
                        if len(k)>=1:
                            sents.append(k+'。')
                            if sentswithlabel.get(i[2])!=None:
                                sentswithlabel.get(i[2]).append(k+'。')
                            else:
                                sentswithlabel[i[2]]=[k+'。']

        self.sents=sents
        self.sentswithlabel= sentswithlabel

        self.sents_statistics()
        self.paragraphs_statistics()
        if sents_to_csv:
            dataframe = pd.DataFrame({ 'sents': sents})
            # 将DataFrame存储为csv,index表示是否显示行名，default=True
            dataframe.to_csv(r'D:\PyProject\docx_input\sents_out_files\\' +
                             self.name + "_sents.csv", index=False, sep=',')

    def setsents_sim(self, param):
        self.sents_sim=param

    def sents_statistics(self):
        #平均句长
        average_sentence_length=0
        sentslen=[]
        for i in self.sents:
            average_sentence_length+=len(i)/len(self.sents)
            sentslen.append(len(i))
        #句长分布
        if config.config.sentences_statistics_show_on:
            print(self.name,)
            print('平均句长',average_sentence_length)
            print('句长',sentslen)

    def paragraphs_statistics(self):
        #平均段内句子数
        average_sentsnum_per_paragraph=0
        average_wordssnum_per_paragraph=0
        paragraphslennum = []#m每一段的句子数目
        paragraphslens=[]
        paragraphslen=0
        for i in self.sentswithlabel.keys():
            #每一个段
            average_sentsnum_per_paragraph += len(self.sentswithlabel.get(i)) / len(self.sentswithlabel.keys())
            paragraphslennum.append(len(self.sentswithlabel.get(i)))
            for sent in self.sentswithlabel.get(i):
                #段中每一个句子
                #平均每段的字长
                average_wordssnum_per_paragraph+=len(sent)/len(self.sentswithlabel.keys())
                paragraphslen+=len(sent)
            paragraphslens.append(paragraphslen/len(self.sentswithlabel.get(i)))

        if config.config.paragraphs_statistics_show_on:
            print(self.name)
            print('平均段内句子数', average_sentsnum_per_paragraph)
            print('平均每段的字长',average_wordssnum_per_paragraph)
            print('每段的句子数目',paragraphslennum)
            print('每段的长度', paragraphslens)
        draw.task1_draw.draw_x_y_distribution(paragraphslennum,paragraphslens,self.name)




