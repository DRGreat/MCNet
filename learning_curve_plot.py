import numpy as np
import proplot as pplt
from matplotlib.pyplot import savefig

def plotLearningCurve(*paths):
    train = []
    val = []
    for path in paths:
        t=[]
        v=[]
        with open(path,"r") as f:
            line = f.readline()
            while line[1] != "t":
                line = f.readline()
            while line:
                if line != "\n":
                    if line[1] == "t":
                        accuracy = float(line.split(":")[-1][0:-2])
                        t.append(accuracy)
                    elif line[1] == "v":
                        accuracy = float(line.split(":")[-1][0:-2])
                        v.append(accuracy)
                line = f.readline()
        train.append(t)
        val.append(v)
    train = np.array(train).transpose()[2:-1,:]
    val = np.array(val).transpose()[2:-1,:]

    fig = pplt.figure(suptitle='Learning curves')
    ax = fig.subplot(xlabel='episode', ylabel='accuracy (%)')
    # ax.plot(train, lw=4, ls = ":")
    # ax.plot(val, lw=1)

 
    ht = ax.plot(
        train, linewidth=1,
        cycle='ggplot',
        labels=["a1","b1"],
        ls=":"
    )

    hv = ax.plot(
        val, linewidth=1,
        cycle='ggplot',
        labels=["a","b"],
        ls="-"
    )
    
    ax.legend([ht,hv], loc='lr')

    fig.savefig("learningcurve")



    


if __name__ == "__main__":
    path1 = "/data/data-home/chenderong/work/renet/log/pretrain vs no-pretrain/miniimagenet_5way1shot_log25124221"
    path2 = "/data/data-home/chenderong/work/renet/log/pretrain vs no-pretrain/miniimagenet_5way1shot_log25124359"
    plotLearningCurve(path1,path2)