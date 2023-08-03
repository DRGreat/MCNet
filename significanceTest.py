from scipy import stats

def dataList(path):
    data = []
    with open(path,"r") as f:
        content = f.readlines()
    for c in content:
        if c.startswith("[test]"):
            data.append(float(c.split(":")[-1]))
    return data

data1 = dataList("/data/data-home/chenderong/work/MCNet/log/more_detail_about_test_episode/cub_5way5shot_log_mcnet")
data2 = dataList("/data/data-home/chenderong/work/MCNet/log/more_detail_about_test_episode/cub_5way5shot_log_renet")


interval = 1999
for i in range(0, 1999):
    if i + interval >= 1999:
        break
    dataslice1 = data1[i:i+interval]
    dataslice2 = data2[i:i+interval]
    if stats.ttest_ind(dataslice1, dataslice2).pvalue < 0.05:
        print(i, interval)
        print(stats.ttest_ind(dataslice1, dataslice2).pvalue)