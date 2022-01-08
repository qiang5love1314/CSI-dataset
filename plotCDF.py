import statsmodels.api as sm
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def read(path):
    sample = loadmat(path)
    sample = np.array(sample['array']).flatten()
    ecdf = sm.distributions.ECDF(sample)
    x = np.linspace(min(sample), max(sample))
    y = ecdf(x)
    return x, y

def main():

    '-----Lab and Meeting Room-----'
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/ILCL-Lab-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/MaoTNSE-Lab-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/Li-TOETICI-Lab-Error.mat')
    # bagging = np.load(r"/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/code/20200722/wing47/bagging.error.npy")
    # ran = np.arange(-0.3, 6, 0.3)
    # cnt4, gap4 = np.histogram(bagging, ran)
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/BLS-Lab-Error.mat')
    # x6, y6 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/SWIM-TMC-Lab-Error.mat')

    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/ILCL-Meet-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/MaoTNSE-Meet-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/Li-TOETICI-Meet-Error.mat')
    # bagging = np.load(r"/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/code/20200722/wing55/p6_bagging.error.npy")
    # ran = np.arange(-0.3, 6, 0.3)
    # cnt4, gap4 = np.histogram(bagging, ran)
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/BLS-Meeting-Error.mat')
    # x6, y6 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/SWIM-TMC-Meet-Error.mat')

    '-----the impact of different antenna-----'
    # x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/不同天线数量/Lab-OneAntenna-Error.mat')
    # x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/不同天线数量/Lab-TwoAntenna-Error.mat')
    # x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/ILCL-Lab-Error.mat')
    # x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/不同天线数量/Meet-OneAntenna-Error.mat')
    # x5, y5 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/不同天线数量/Meet-TwoAntenna-Error.mat')
    # x6, y6 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/ILCL-Meet-Error.mat')

    '-----CDF of Incremental learning-----'
    x1, y1 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/增量式学习性能/ILCL-Lab-Error-Incremental.mat')
    x2, y2 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/增量式学习性能/ILCL-Lab-Retrain-Error.mat')
    x3, y3 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/增量式学习性能/ILCL-Meet-Error-Incremental.mat')
    x4, y4 = read(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/增量式学习性能/ILCL-Meet-Retrain-Error.mat')

    sample = loadmat(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/增量式学习性能/ILCL-Meet-Retrain-Error.mat')
    sample = np.array(sample['array']).flatten()
    print(np.mean(sample))
    print(np.std(sample))

    figure, ax = plt.subplots()
    'comparison'
    # plt.step(x1, y1, color = 'blue', marker ='.', label='ILCL')
    # plt.step(x2, y2, color='green', marker='v', label='CiFi')
    # plt.step(x3, y3, color='red', marker='x', label='AF-DCGAN')
    # plt.step(gap4[1:], np.cumsum(cnt4) / sum(cnt4), color='c', marker='^', label='EnsemLoca')
    # plt.step(x5, y5, color='m', marker='p', label='BLS-Location')
    # plt.step(x6, y6, color='orange', marker='d', label='SWIM')

    '-----the impact of different antenna-----'
    # plt.step(x1, y1, color = 'blue', marker ='.', label='One antenna-NLOS')
    # plt.step(x2, y2, color='green', marker='v', label='Two antennas-NLOS')
    # plt.step(x3, y3, color='red', marker='x', label='Three antennas-NLOS')
    # plt.step(x4, y4, color='c', marker='^', label='One antenna-LOS')
    # plt.step(x5, y5, color='m', marker='p', label='Two antennas-LOS')
    # plt.step(x6, y6, color='orange', marker='d', label='Three antennas-LOS')

    '-----CDF of Incremental learning-----'
    plt.step(x1, y1, color='blue', marker='.', label='ILCL-NLOS')
    plt.step(x2, y2, color='green', marker='v', label='Retraining-NLOS')
    plt.step(x3, y3, color='red', marker='x', label='ILCL-LOS')
    plt.step(x4, y4, color='c', marker='^', label='Retraining-LOS')


    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Estimated Errors (m)',font2)
    plt.ylabel('CDF',font2)
    plt.grid(color="grey", linestyle=':', linewidth=0.5)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc = 'lower right')
    # plt.savefig('CDF_Incremental_Accuracy.pdf', bbox_inches = 'tight', dpi=500)
    plt.show()

if __name__ == '__main__':
    main()
    pass