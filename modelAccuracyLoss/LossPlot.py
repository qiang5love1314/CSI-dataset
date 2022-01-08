import matplotlib.pyplot as plt
import pickle

def loadHistory(path):
    with open(path,'rb') as file_txt:
        history = pickle.load(file_txt)
    return history

path1 = '/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/modelAccuracyLoss/logLab.txt'
path2 = '/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/modelAccuracyLoss/logMeet.txt'

history1 = loadHistory(path1)
history2 = loadHistory(path2)

figure, ax = plt.subplots()
plt.plot(history1['loss'], color = '#5c81e2', linewidth =2, label='Lab')
plt.plot(history2['loss'], color = 'orange', linestyle = '--', linewidth =2, label='Meeting Room')

font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
plt.xlabel('Epoch',font2)
plt.ylabel('Training Loss',font2)

plt.grid(color="grey", linestyle=':', linewidth=0.5)
plt.tick_params(labelsize=15)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.legend(prop={'family': 'Times New Roman', 'size': 12}, loc = 'upper right')
# plt.savefig('Training_Loss.pdf', bbox_inches = 'tight', dpi=500)
plt.show()


