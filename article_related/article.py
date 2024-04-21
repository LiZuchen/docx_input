# -*- coding: utf-8 -*-
import re
import os
import config.config
import draw.task1_draw
import matplotlib
import pandas as pd
import seaborn
import sns as sns
from article_related.reference import reference
from bert_embedding.bert_emb import cal_ppl_bygpt2, cal_sim_bybert
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode
from draw.task1_draw import pplnum_drawbar
import pickle
import bibtexparser
from sympy import stats
import matplotlib.pyplot as plt

class article:
    def __init__(self, name, paragraphs):
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
        print('ppl_check_in '+self.name)
        if config.config.ppl_show_in_console:
            for i,j in zip(self.ppllist,self.sents):
                if i>50:
                    print(i,j)
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

        if config.config.sents_sim_check_on:
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
        # sents_to_csv=config.config.sents_to_csv
        if config.config.sents_cache_on ==0:
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
            dataframe = pd.DataFrame({'sents': sents})
            # 将DataFrame存储为csv,index表示是否显示行名，default=True
            dataframe.to_csv(r'D:\PyProject\docx_input\sents_out_files\\' +
                             self.name + "_sents.csv", index=False, sep=',')
            self.sents = sents
            with open(r'D:\PyProject\docx_input\sents_out_files\\' +
                             self.name + "_sentswithlabel.pkl", "wb") as tf:
                pickle.dump(sentswithlabel, tf)
                tf.close()
            # # 读取文件
            # with open("myDictionary.pkl", "rb") as tf:
            #     new_dict = pickle.load(tf)

            # with open(r'D:\PyProject\docx_input\sents_out_files\\' +
            #                  self.name + "_sentswithlabel.txt",'w') as f:
            #     print(sentswithlabel,f)
                self.sentswithlabel = sentswithlabel
        else:
            x=pd.read_csv(r'D:\PyProject\docx_input\sents_out_files\\' +
                             self.name + "_sents.csv",sep=',')
            self.sents = list(x['sents'])

            with open(r'D:\PyProject\docx_input\sents_out_files\\' +
                      self.name + "_sentswithlabel.pkl", "rb") as tf:
                sentswithlabel=pickle.load(tf)


            # with open(r'D:\PyProject\docx_input\sents_out_files\\' +
            #                  self.name + "_sentswithlabel.txt", "r", encoding='gbk') as f:  # 打开文件
            #     data = f.read()  # 读取文件
                    # print(data)
                self.sentswithlabel=sentswithlabel



        self.sents_statistics()
        self.paragraphs_statistics()


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
        self.average_sentence_length=average_sentence_length
        self.sentslen=sentslen
        save=[average_sentence_length,sentslen]
        with open ('D:\PyProject\docx_input\data_cache\sents_para_save\\'+self.name+'_s_save.txt','w') as f:
            print(save,file=f)
        f.close()

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
        self.average_sentsnum_per_paragraph=average_sentsnum_per_paragraph
        self.average_wordssnum_per_paragraph=average_wordssnum_per_paragraph
        self.paragraphslennum=paragraphslennum
        self.paragraphslens=paragraphslens
        save=[average_sentsnum_per_paragraph,average_wordssnum_per_paragraph,paragraphslennum,paragraphslens]
        with open ('D:\PyProject\docx_input\data_cache\sents_para_save\\'+self.name+'_p_save.txt','w') as f:
            print(save,file=f)
        f.close()
        if config.config.task1_3_draw_on:
            draw.task1_draw.draw_x_y_distribution(paragraphslennum,paragraphslens,self.name)

    def getref(self):
        return self.refs
    def setref(self):
        c=0
        b=0
        refs=[]
        for i in self.text:
            if i=='参考文献' and c==0:
                c+=1
                b=1
                continue
            elif i=='参考文献':
                print('error in getref while processing article',self.name)
            if '.' not in i and b==1:
                break
            if c>=1 and b==1:
                refi=reference(self,c,i)
                refs.append(refi)
                c+=1
        self.refs = refs

        if self.refs[0].text[0] != '[':
            x=1
            for i in self.refs:
                i.text='['+str(x)+']'+i.text
                x+=1

        # er_unicode=[]
        fail = []
        for ref in self.refs:
            key = ref.getusera()
            ref.setkey(key)
            if key == None:
                fail.append(ref.text)
                ref.no_key=True
        if config.config.ref_write_file_on ==1:
            filepath = r'D:\PyProject\docx_input\data_cache\refs\\' + self.name + '.txt'
            for ref in self.refs:
                if ref.key!=None:
                    with open(filepath,'a',encoding='utf-8') as f:  # 打开文件
                    # try:
                        print(ref.text, file=f)
        if config.config.refs_parse_cache_on ==0:
            in_path=r'D:\PyProject\docx_input\data_cache\refs\\' + self.name + '.txt'
            out_path=r'D:\PyProject\docx_input\data_cache\refs_bibtex\\' + self.name + '.bib'
            log_path=r'D:\PyProject\docx_input\data_cache\refs_log\\' + self.name + '.txt'
            os.system('perl "d:\PyProject\docx_input\gb7714\gb7714texttobib.pl" '
                      'in='+in_path+
                      ' out='+out_path+
                      ' log='+log_path)
            with open(r'D:\PyProject\docx_input\data_cache\refs_fail\\' + self.name + '_nokey.txt', 'w', encoding='utf-8') as f:  # 打开文件
                # try:
                print(fail, file=f)
        bibpath=r'D:\PyProject\docx_input\data_cache\refs_bibtex\\' + self.name + '.bib'
        with open(bibpath,encoding='UTF-8',errors='ignore') as bibtex_file:
            parser = BibTexParser(ignore_nonstandard_types=True)  # 声明解析器类
            parser.customization = convert_to_unicode  # 将BibTeX编码强制转换为UTF编码
            bibdata = bibtexparser.load(bibtex_file, parser=parser)  # 通过bp.load()加载
        self.refs_entries=bibdata.entries
        self.ref_titles=[]
        for entry in self.refs_entries:
            # 获取文章的标题
            try:
                self.ref_titles.append(entry['title'])
            # 获取文章的作者
            except KeyError:
                print(entry)
            # 获取文章的发表年份
        self.setref_date()
    # print(bib_database.entries)
        self.setcite()

        return self.refs

    def setref_date(self):
        ref_date=[]

        indexj=0
        for entry in self.refs_entries:
            if 'date'in entry.keys():
                if  entry['title'] != None:
                    try:
                        year=int(entry['date'][:4].replace(" ",""))
                        if year>2024 or year<1900:
                            print('date error in',entry['title'],self.name)
                        else:
                            ref_date.append(year)
                            # while self.refs[indexj].no_key==True :
                            #     indexj+=1

                                #self.refs
                                #index为bibetex的索引号
                            # self.refs[indexj].date = year
                            # print(self.refs[indexj].date, self.refs[indexj].text)
                            # indexj+=1
                    except ValueError:
                        print(entry['date'][:4])
            else:
                indexj += 1
        ref_date.sort()
        self.ref_date_statistic(ref_date)
        print(ref_date)
    def ref_date_statistic(self,dates):
        dlf=[]

        c=0
        for i in self.refs:
            # print(i)
            dl = []
            for si in re.findall('[0-9]+',i.text):
                if int(si)>1920 and int(si)<2025:
                   dl.append(int(si))
            ds=set(dl)
            if len(ds)==1:
                if i.date!=None:
                    if i.date in ds:
                        dlf.append(i.date)
                    else:
                        print('find not the same date record in .date')
                        print(i.date,print(ds))
                else:
                    i.date=list(ds)[0]
                    dlf.append(i.date)

            elif len(ds)==0:
                #ref没有date
                pass
            else:
                print('存在两个以上的日期，ds = ',ds,end=' ')
                if i.date in ds:
                    print('能找到一个')
                    dlf.append(i.date)
                else:
                    print('error in find only year，存在多个可能日期,不进行统计')
                    print(i.date,i.text)

        dlf.sort()
        if c==1:
            earliest=dates[0]
            latest=dates[-1]
            average=sum(dates)/len((dates))
            dateset=set(dates)
            datedict={}
            for d in dateset:
                datedict[d]=dates.count(d)
            print(datedict)
            draw.task1_draw.refdate_dis_drawbar(datedict, self.name)
        else:
            earliest = dlf[0]#最久远的参考文献
            latest = dlf[-1]#最近的参考文献
            average = sum(dlf) / len((dlf))
            datedict = {}
            for d in set(dlf):
                datedict[d] = dlf.count(d)
            print(datedict,'f:',earliest,'l:',latest,'a:',average)
            draw.task1_draw.refdate_dis_drawbar(datedict, self.name)
        return
    def show_ref_cite(self):
        for i in self.refs:
            print(i.text)
            for c in i.getcontext():
                print(c)
    def setcite(self):
        for i in self.sents:
            #每一句
            if i[0]!='[':#不是参考文献
                res=re.findall(r'\[[ 0-9]+]',i)#提取句中引用编号

                if len(res)>0:
                    itero=[]
                    for j in res:#对每一个句中的引用编号查找该编号对应的参考文献--->self.refs[int(j[1:-1]) - 1]
                        if ' ' in j:
                            for k in list(filter(None,re.split(' |\[|]',j))):
                                # print(k)
                                itero.append(int(k))
                        else:
                            itero.append(int(j[1:-1]))
                    itero=list(set(itero))
                    for l in itero:
                        if self.refs[l - 1].index!=l or l - 1<0:
                            pass
                            # print(l - 1)
                            # print('ref_getcite_check error!',self.name)
                            # print('----------------------------------')
                            # print(self.refs[l - 1].text)
                            # print(i)
                            # print(self.refs[l - 1].index)  # index为引用序号
                            # print('----------------------------------')
                        else:
                            # if len(self.refs[l - 1].getcontext())==0:
                            #     print('first context:')
                            #
                            # else:
                            #     print('more context!')
                            #
                            # print(i)
                            self.refs[l- 1].addcontext(i)
        # self.show_ref_cite()
        # self.ref_cite_check()
    def ref_cite_check(self):

        sim=[]
        i=0
        for ref in self.refs:
            if ref.no_key:
                print(ref.text)
            else:
                print(ref.text)
                print(self.ref_titles[i])
                sim.append(ref.text_context_sim(self.ref_titles[i]))
                i += 1
        print(sim)

