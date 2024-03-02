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

    def setparagraphs(self,paragraphs):
        self.paragraphs=paragraphs
    def getparagraphs(self):
        return self.paragraphs
