# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from matplotlib.ticker import MultipleLocator
from seaborn import jointplot


def pplnum_drawbar(pplnum,name):
    matplotlib.use('TkAgg')
    n=len(pplnum)
    x=[2,4,6,8,10,12,14,16]#确定柱状图数量,可以认为是x方向刻度

    y=pplnum#y方向刻度

    color=['green','yellow','orange','brown','red','purple','blue','black']
    x_label=['0-10','10-25','25-50','50-100','100-500','500-1000','1000-2000','>2000']
    # plt.xticks(x, x_label)#绘制x刻度标签
    # plt.bar(x, y,width=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],bottom=0,color=color)#绘制y刻度标签
    # plt.legend()
    # plt.grid(True, linestyle=':', color='r', alpha=0.6)
    # plt.show()

    fig, ax = plt.subplots(dpi=200,figsize=(30,10))
    ax.bar(x, y,color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(x_label)
    plt.tick_params(labelsize=25)
    # plt.xlabel(labelpad=0.5)
    # width=0.2
    # x_major_locator = MultipleLocator(1)
    # ax.xaxis.set_major_locator(x_major_locator)
    # ax.set_xlim([-0.5, 2.5])
    plt.title(name,fontproperties="STSong",fontsize=40)

    figpath='D:\PyProject\docx_input\data_cache\ppl_distribution_figure'
    plt.savefig(figpath+'\\'+name+'.png')
    # plt.show()
    plt.close()
    #设置网格刻度
def draw_x_y_distribution(x,y,name):
    matplotlib.use('TkAgg')
    with seaborn.axes_style("white"):
        # fig, ax = plt.subplots(figsize=(20, 20))
        # 创建一个六边形蜂窝图联合图
        jointplot(x=x, y=y, kind="hex").set_axis_labels(
            xlabel="段落句子数分布", ylabel="段落长度分布", fontproperties='STsong')
    plt.tight_layout()
    figpath = 'D:\PyProject\docx_input\data_cache\paragraphs_distribution'
    plt.savefig(figpath + '\\' + name +'paragraphs__distribution_figure' +'.png')
    # plt.show()

def tmp():
    matplotlib.use('TkAgg')
    x=[1,2,3,4,5,6,7,8,9,10]
    y1=[1055.43,1213.67,1208.42,1211.61 ,1207.69,1182.04,1171.39,630.28,643.06,663.76]
    y2=[159.09,315.44 ,470.71 ,484.96 ,497.23 ,470.88 ,461.04 ,470.93 ,463.38 ,474.23]
    y3=[ 265.42 ,316.89 ,316.77,317.89,343.91,337.73,312.40 ,315.34 ,333.13 ,321.91 ]
    fig, ax = plt.subplots(dpi=100, figsize=(10, 10))
    ax.plot(x,y3)

    plt.xlabel('并行数量',fontproperties="STSong")
    plt.ylabel('RPS',fontproperties="STSong")
    plt.show()


def refdate_dis_drawbar(data,name):
    #data-->dict
    matplotlib.use('TkAgg')
    n=len(data)
    x=[]
    y=[]
    for i in sorted(data.keys()):
        x.append(i)
        y.append(data[i])

    # plt.xticks(x, x_label)#绘制x刻度标签
    # plt.bar(x, y,width=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5],bottom=0,color=color)#绘制y刻度标签
    # plt.legend()
    # plt.grid(True, linestyle=':', color='r', alpha=0.6)
    # plt.show()

    fig, ax = plt.subplots(dpi=200,figsize=(30,10))
    ax.bar(x, y)
    ax.set_xticks(x)
    plt.tick_params(labelsize=10)
    plt.title(name+'参考文献年份分布',fontproperties="STSong",fontsize=40)
    ax.yaxis.set_major_locator(MultipleLocator(1))
    figpath=r'D:\PyProject\docx_input\data_cache\refdate_distribution_figure'
    plt.savefig(figpath+'\\'+name+'.png')
    # plt.show()
    plt.close()
