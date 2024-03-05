# -*- coding: utf-8 -*-
import os
import time

from article_related.article import article
from config.config import readpathdir
from docx import Document
from input.block_divide import blockdivide
from input.readsentences import zykeywordextract, menuextract
from input.rwsinput import rws

def article_build():

    files=os.listdir(readpathdir)
    articlelist=[]
    for file in files:
        print(file)
        file_path = os.path.join(readpathdir, file)
        doc = Document(file_path)
        # print(file_path)
        # print(len(doc.paragraphs))
        name = file.split('.')[0]
        x=article(name,doc.paragraphs)
        # x.setname(name)
        # x.setparagraphs(doc.paragraphs)
        articlelist.append(x)
    #rws build
    for a in articlelist:
        a.setrws(rws(a.getparagraphs(),a.getname()))#任务书
        res=zykeywordextract(a.getparagraphs(),a)#摘要和关键词
        a.setzy(res[0])
        a.setkeyword(res[1])
        a.setmenu(menuextract(a.getparagraphs(),a))#目录
        a.setblocks(blockdivide(a))#按目录分块
        a.menucheck()#目录检测
        a.blockstocsv()#保存块
        a.tosents()#生成句子（如果已有，则直接读取填充）
        a.calppl()#计算ppl，如果已经计算，则直接填充
        a.cal_sents_sim()#计算句子上下句相似度，如果已经计算，则直接填充
        a.getref()
    return articlelist
bgtime=time.time()
articlelist=article_build()
print('article_build() 用时： ',time.time()-bgtime,' s')
# articlelist[0].blockstocsv()
# articlelist[0].calppl()
# blockdivide(articlelist[3])
# articlelist[0].pplcheck()
m=1