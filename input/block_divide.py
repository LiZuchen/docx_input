# -*- coding: utf-8 -*-
def blockdivide(a):
    print(a.getname())
    blocks=[]
    hdnum=[]
    #menuitem===>text ,heading x, pi
    for hd in a.getmenu():
        hdnum.append(hd[2])#hd[2]--->pi

    for num in range(0,len(hdnum)):
        blocklist=[]
        if num<len(hdnum)-1:
            for i in a.getparagraphs()[hdnum[num]:hdnum[num+1]]:
                # print(i.text)
                blocklist.append(i.text)
        else:
            for i in a.getparagraphs()[hdnum[num]:len(a.getparagraphs())]:
                # print(i.text)
                blocklist.append(i.text)
        print("----------------------------------------------------------------------")
        blocks.append([a.getmenu()[num][0],a.getmenu()[num][1],a.getmenu()[num][2],blocklist])
    return blocks