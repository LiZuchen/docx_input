import argparse
import re
from collections import defaultdict
import bibtexparser
from bert_embedding.bert_emb import calc_similarity


class reference:

    def __init__(self, article, index,text):
        self.article=article
        self.index=index
        self.text=text
        self.context=[]
        self.no_key=False

    def setkey(self,key):
        self.key=key
    def getusera(self):
        s=self.text
        s = re.sub(r'EB/OL', r'EB', s)
        res = re.findall(r'\[(.*?)\]', s)
        useras = {
            'J': ['author.title[usera].translator,year,volume(number):pages[urldate].url.doi.',
                  'author.title[usera].translator,year,volume(number):pages.url.',
                  'author.title[usera].translator,year,volume(number):pages.',
                  'author.title[usera].translator,volume(number):pages.',
                  'author.title[usera].translator,year,volume:pages.',
                  'author.title[usera].translator,year,volume(number):pages.doi.',
                  'author.title[usera].translator,year:pages.doi.'],

            'J/OL': ['author.title[usera].translator,year,volume(number):pages[urldate].url.doi.',
                     'author.title[usera].translator,year,volume(number):pages.url.',
                     'author.title[usera].translator,year,volume(number):pages.'],
            'M': ['author.title[usera].location:publisher.',
                  'author.title[usera].location:publisher,year:pages,volume.',
                  'author.title[usera].publisher,year:pages.'],
            'R': ['author.title[usera].location:publisher,date.',
                  'author.title:subtitle[usera].location:publisher,date.'],
            'EB': ['author.title[usera].url,date.',
                   'author.title[usera].url,date/urldate.', ],
            'EB/OL': ['author.title[usera].url,date.',
                      'author.title[usera].url,date/urldate.',
                      'title[usera].url.'],
            'N': ['author.title[usera].journaltitle,date(number).', ],
            'D': ['author.title[usera].address:publisher,year.',
                  'author.title[usera].address,year.'],
            'C': ['author.title.edition[usera].location:publisher,year:pages.'],
            'P': ['author.title.edition[usera].location:publisher,year:pages.',
                  'author.title.edition[usera].date.']
        }
        if res is not None:
            for usera in res:
                if usera in useras.keys():
                    res = usera
                    return res
        else:
            print(s,'get usera none')

    def toTemplate(self):
        useras = {
            'J': ['author.title[usera].translator,year,volume(number):pages[urldate].url.doi.',
                  'author.title[usera].translator,year,volume(number):pages.url.',
                  'author.title[usera].translator,year,volume(number):pages.',
                  'author.title[usera].translator,volume(number):pages.',
                  'author.title[usera].translator,year,volume:pages.',
                  'author.title[usera].translator,year,volume(number):pages.doi.',
                  'author.title[usera].translator,year:pages.doi.'],

            'J/OL': ['author.title[usera].translator,year,volume(number):pages[urldate].url.doi.',
                  'author.title[usera].translator,year,volume(number):pages.url.',
                  'author.title[usera].translator,year,volume(number):pages.'],
            'M': ['author.title[usera].location:publisher.',
                  'author.title[usera].location:publisher,year:pages,volume.',
                  'author.title[usera].publisher,year:pages.'],
            'R': ['author.title[usera].location:publisher,date.',
                  'author.title:subtitle[usera].location:publisher,date.'],
            'EB': ['author.title[usera].url,date.',
                   'author.title[usera].url,date/urldate.', ],
            'EB/OL': ['author.title[usera].url,date.',
                   'author.title[usera].url,date/urldate.',
                      'title[usera].url.'],
            'N': ['author.title[usera].journaltitle,date(number).', ],
            'D': ['author.title[usera].address:publisher,year.',
                  'author.title[usera].address,year.'],
            'C': ['author.title.edition[usera].location:publisher,year:pages.'],
            'P': ['author.title.edition[usera].location:publisher,year:pages.',
                  'author.title.edition[usera].date.'],
            'Z':['author.title[usera].translator,year,volume(number):pages[urldate].url.doi.']
        }
        entrytypes = {
            'M': 'book',
            'J': 'article',
            'C': 'proceedings',
            'G': 'collection',
            'N': 'newspaper',
            'D': 'mastersthesis',
            'R': 'report',
            'EB': 'online',
            'S': 'standard',
            'P': 'patent',
            'DB': 'database',
            'CP': 'software',
            'A': 'archive',
            'CM': 'map',
            'DS': 'dataset',
            'Z': 'misc',
        }
        for usera in useras:
            useras[usera].sort(key=len, reverse=True)

        def str2pattern(s):
            pattern = s
            pattern = re.sub(r'\.', r'\.', pattern)
            pattern = re.sub(r'\[', r'\[', pattern)
            pattern = re.sub(r'\]', r'\]', pattern)
            pattern = re.sub(r'\(', r'\(', pattern)
            pattern = re.sub(r'\)', r'\)', pattern)
            pattern = re.sub(r'\w+', '(.*?)', pattern)
            files = re.findall(r'\w+', s)
            return pattern, files

        def delspace(s):
            s = re.sub(r'^ *\[\d+\] *', r'', s.strip())
            s = re.sub(r' *([\.,\[\]\(\):]) *', r'\1', s)
            return s

        def getusera(s):
            s = re.sub(r'EB/OL', r'EB', s)
            res = re.findall(r'\[(.*?)\]', s)

            if res is not None:
                for usera in res:
                    if usera in useras.keys():
                        res = usera
                        return res
            else:
                print(s,'get usera none')
            # raise KeyError(
            #     'Could not find pattern for usera ' + str(res) + ', plese add it manually to useras dictionary.')

        def parse(s):
            parsed_dict = defaultdict(str)
            s = delspace(s)
            parsed_dict['source'] = s
            # print("parsing", s)
            usera = getusera(s)
            styles = useras.get(usera)
            if styles is None:
                # raise IndexError("Couldn't find pattern of entry file " + usera)
                print("Couldn't find pattern of entry file ")
                print(s)
            else:
                for style in styles:
                    pattern, files = str2pattern(style)
                    res = re.match(pattern, s)
                    if res is not None:
                        res = res.groups()
                        assert len(files) == len(res)
                        parsed_dict.update(dict(zip(files, res)))
                        parsed_dict['author'] = re.sub(r',', r' and ',
                                                       parsed_dict['author'])  # use 'and' to seperate authors
                        parsed_dict['usera'] = usera
                        # print("matched!", style)
                        # print(len(parsed_dict) - 1, "entry files has found!")
                        # print(s)
                        return parsed_dict
                print(s)

            return None
        # parser = argparse.ArgumentParser(
        #     description='Parse GB/T 7714 into bibtex style.')
        # parser.add_argument("input",
        #                     metavar="<input>",
        #                     help="input GB/T 7714 filename")
        # parser.add_argument("output",
        #                     metavar="<output>",
        #                     help="output bibtex filename")
        # args = parser.parse_args()

        failed = list()
        # with open(args.input, 'r') as f:
        #     lines = f.readlines()
        # o = open(args.output, 'w')
        # for line in lines:
        if self.text[-1]!='.':
            self.text=self.text+'.'
        parsed_dict = parse(self.text)
        # print(parsed_dict)
            # if parsed_dict is not None:
            #     o.write(genbib(parsed_dict))
            # else:
            #     failed.append(line)
        # o.close()
        # print("writed to", args.output)

        if len(failed) > 0:
            print("failed to parse:")
            for i in failed:
                print('\t', i)

        return parsed_dict!=None
    def addcontext(self, context):
        self.context.append(context)
    def getcontext(self):
        return self.context
    def text_context_sim(self,title):
        res=[]
        for i in self.context:
            res.append(calc_similarity(title,i))
        return res

            # 获取文章的作者
            # 获取文章的发表年份
            # print(title)
        # if not self.no_key:
        #     print(self.article.refs_entries[self.index-1])

        # print(bib_database.entries)

