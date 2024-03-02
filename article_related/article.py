# -*- coding: utf-8 -*-
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

