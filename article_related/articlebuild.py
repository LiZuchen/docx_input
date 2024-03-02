# -*- coding: utf-8 -*-
import os

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
        print(file_path)
        print(len(doc.paragraphs))
        name = file.split('.')[0]
        x=article()
        x.setname(name)
        x.setparagraphs(doc.paragraphs)
        articlelist.append(x)
    #rws build
    for a in articlelist:
        a.setrws(rws(a.getparagraphs(),a.getname()))#任务书
        res=zykeywordextract(a.getparagraphs(),a)#摘要和关键词
        a.setzy(res[0])
        a.setkeyword(res[1])
        a.setmenu(menuextract(a.getparagraphs(),a))#目录
        a.setblocks(blockdivide(a))
    return articlelist
articlelist=article_build()
blockdivide(articlelist[3])
m=1