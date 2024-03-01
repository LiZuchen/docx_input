import os

from config.config import readsample
from config.config import readpathdir
from docx import Document
import config
def zyout(path,text):
    with open(path, 'w') as f:
        print(text,file=f)
file_path=readsample
doc = Document(file_path)
paragraphs=doc.paragraphs
print(file_path)
print(paragraphs)
for p in paragraphs:
    print(p.text)

files=os.listdir(config.config.readpathdir)
for file in files:
    print(file)
    name=file.split('.')[0]
    file_path = os.path.join(config.config.readpathdir, file)
    doc = Document(file_path)
    pi=0
    fzyi=0
    for i in doc.paragraphs:
        xi=list(filter(None,i.text.split(" ")))
        pi+=1
        if len(xi)>=1 and xi[0]=='摘要':
            print(file)
            print(xi[0])
            fzyi=pi
        elif len(xi)>=2 and xi[0]=='摘' and xi[1]=='要':
            print(file)
            print(xi[0])
            print(xi[1])
            fzyi = pi
        if len(xi)>=1and len(xi[0]) >= 3 and xi[0][0:3] == '关键词':
            print(xi[0])
            break
    zytext=[]
    keywordtext=[]
    for pit in doc.paragraphs[fzyi:pi-1]:
        #每个pit.text为摘要中的一段话
        print(pit.text)
        zytext.append(pit.text)
    try :
        keywordstr=doc.paragraphs[pi-1].text.split('：')[1]
    except:
        print("关键词部分不由中文冒号分隔")

    if '，' in keywordstr:
        keywordtext = list(keywordstr.split('，'))
    elif '、' in keywordstr:
        keywordtext = list(keywordstr.split("、"))
    else:
        print("------------------------")
        print("关键词不由中文 , or 、分隔")
        print("------------------------")
    # except:
    #     print("关键词不由 , or 、分隔")

    zyout('D:\PyProject\docx_input\zy_output_files\\'+name+'.txt',zytext)
    zyout('D:\PyProject\docx_input\keyword_output_files\\'+name+'.txt',keywordtext)