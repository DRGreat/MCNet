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
    train = np.array(train).transpose()
    val = np.array(val).transpose()

    fig = pplt.figure()
    ax = fig.subplot(xlabel='episode', ylabel='accuracy (%)')
    # ax.plot(train, lw=4, ls = ":")
    # ax.plot(val, lw=1)

 
    ht = ax.plot(
        train, linewidth=1,
        cycle='ggplot',
        labels=["OURS /train","RENet /train"],
        ls="--"
    )

    hv = ax.plot(
        val, linewidth=1.5,
        cycle='ggplot',
        labels=["OURS /val","RENet /val"],
        ls="-"
    )
    
    ax.legend(ht+hv, loc='top',ncol=4)

    fig.savefig("learningcurve")



    


if __name__ == "__main__":

    methodp = "/data/data-home/chenderong/work/renet/log/ablation/cat[7,8]/cub_5way1shot_log14201813"
    RENetp = "/data/data-home/chenderong/work/renet/log/baseline/cub_5way1shot_log09140744"
    plotLearningCurve(methodp, RENetp)