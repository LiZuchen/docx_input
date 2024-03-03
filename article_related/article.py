# -*- coding: utf-8 -*-
import re

import config.config
import pandas as pd
from bert_embedding.bert_emb import cal_ppl_bygpt2
from draw.ppldraw import pplnum_drawbar


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
        cache_on=config.config.ppl_cache_on
        draw_on=config.config.ppl_distribution_draw
        ppl_len_check_on=config.config.ppl_len_check_on
        #如果on，那么要设置一次ppl
        if cache_on:
            filepath=r'D:\PyProject\docx_input\data_cache\ppllist_cache\\'+self.name+'.txt'
            with open(filepath, "r",encoding='gbk') as f:  # 打开文件
                data = f.read()  # 读取文件
                # print(data)
            strlist=list(filter(None,re.split(' |,|\[|]|\n',data)))
            floatlist=list(map(float,strlist))
            self.setppllist(floatlist)
        if ppl_len_check_on:
            if len(self.ppllist) == len(self.text) :
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
            elif i>=500 and i<1000:
            # 6
                y[5] += 1
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
        if config.config.ppl_cache_on:
           self.pplcheck()
        else:
            self.setppllist(cal_ppl_bygpt2(self.text,self.name))
            self.pplcheck()



