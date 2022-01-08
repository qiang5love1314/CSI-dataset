import os
import numpy as np
import pandas as pd
import time
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
from scipy import signal
from numpy import random
from sklearn.preprocessing import LabelBinarizer, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from BroadLearningSystem import *
import pickle
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.preprocessing.image import img_to_array
from keras.callbacks import EarlyStopping
from keras import regularizers, optimizers
import cv2

'CNN processing'
def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def getXlabel():
    xLabel = []
    for i in range(21):     #横坐标
        str = '%d' % (i + 1)
        xLabel.append(str)
    return xLabel

def getYlabel():
    yLabel = []
    for j in range(23):     #纵坐标
        if(j<9):
            num=0
            str= '%d%d' % (num, j+1)
            yLabel.append(str)
        else:
            yLabel.append('%d'% (j+1) )
    return yLabel

def rawCSI():
    xLabel = getXlabel()
    yLabel = getYlabel()
    count = 0
    originalCSI=np.zeros((317, 135000), dtype=np.float)
    newName = []
    label = np.empty((0, 2), dtype=np.int)

    for i in range(21):
        for j in range(23):
            filePath = r"/Users/zhuxiaoqiang/Downloads/47SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
            name = xLabel[i] + yLabel[j]
            if (os.path.isfile(filePath)):
                c = loadmat(filePath)
                CSI = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                originalCSI[count, :] = CSI[:, 3 * 30 * 900]
                newName.append(name)
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                count += 1
    return originalCSI, label

def MatrixToImage(data):
    from PIL import Image
    data = data*255
    new_im=Image.fromarray(data.astype(np.uint8))
    return new_im

def generatePhaseImage():   #已完成，不再重新生成图片
    midString = []
    xLabel = getXlabel()
    yLabel = getYlabel()

    for i in range(21):
        for j in range(23):
            filePath = r'/Users/zhuxiaoqiang/Downloads/thirdDataSet/47imaginary_part/imaginary' + xLabel[i] + yLabel[j] + '.mat'
            if (os.path.isfile(filePath)):
                data = loadmat(filePath)
                test = data['myData']  # 3*30*1500
                for k in range(50):
                    swap = test[:, :, 30 * (k):30 * (k + 1)]
                    newd = np.rollaxis(swap, 0, 3)  # change channel, 30*30*3
                    phase = abs(np.array(newd))  #
                    new_im = MatrixToImage(phase)
                    saveName = r'/Users/zhuxiaoqiang/Downloads/thirdDataSet/47imaginary_part/image-phase-3D/Phase' + xLabel[i] + yLabel[j] +"-"+midString[k]+ ".jpg"
                    # new_im.save(saveName, quality=95, subsampling=0)

def phaseMatrix():
    xLabel = getXlabel()
    yLabel = getYlabel()
    data = []
    phaseLabel = []
    midString = [] #str(list(range(1,50)))
    numOfImage = 30
    for i in range(numOfImage):
        str = '%d' % (i + 1)
        midString.append(str)

    for i in range(21):
        for j in range(23):
            for k in range(numOfImage):
                filePath = r"/Users/zhuxiaoqiang/Downloads/thirdDataSet/47imaginary_part/image-phase-3D/Phase" + xLabel[i] + yLabel[
                           j] + "-" + midString[k] + ".jpg"
                if (os.path.isfile(filePath)):
                    image = cv2.imread(filePath)
                    image = cv2.resize(image,(30,30))
                    image = img_to_array(image)
                    data.append(image)

    count = 0
    for i in range(21):
        for j in range(23):
            filePath = r"/Users/zhuxiaoqiang/Downloads/47SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                count += 1
                phaseLabel.append(numOfImage*[count])
    phaseLabel = np.reshape(phaseLabel, (317*numOfImage, 1)).flatten()
    return data, phaseLabel

def indexOfLabel(Y, label):# 获取标签的索引(正太分布的随机数)
    index = []
    for i in range(len(label)):
        index1 = np.where(Y[:, 0] == label[i][0])
        index2 = np.where(Y[:, 1] == label[i][1])
        similar = list(set(index1[0]).intersection(set(index2[0])))
        index.append(similar)
    return np.array(index).flatten()

def splitDataSet(data, Y, trainRawlabel): #未用
    index = indexOfLabel(Y, trainRawlabel)
    numOfImage = 50
    target = np.zeros((len(trainRawlabel) * numOfImage, 30, 30, 3), dtype=np.float)
    newLabel = []
    for i in range(len(trainRawlabel)):
        start = numOfImage * index[i]
        end = numOfImage * index[i] + numOfImage
        swap = data[start:end]
        target[(i * numOfImage):(i * numOfImage + numOfImage)] = swap
        newLabel.append(numOfImage * [index[i]])
    newLabel = np.reshape(newLabel, (len(trainRawlabel) * numOfImage, 1)).flatten()
    return target, newLabel

def constructCNN(traindata, testdata, trainlabel, testlabel):
    inputShape = (30, 30, 3)
    chanDim = -1

    'conv > relu > pool'
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    #'(conv > relu) * 2 > pool'    , kernel_regularizer=regularizers.l2(0.01)
    model.add(Conv2D(64, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(64, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    '(conv > relu) * 2 > pool'
    model.add(Conv2D(128, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Conv2D(128, (3, 3), padding='SAME', input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    'FC > rely'
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Dropout(0.25))

    'classier > softmax'
    model.add(Dense(317, activation='softmax'))

    'sgd > compile'
    print("compiling model...")
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

    # 'Image enhancement'
    # from keras.preprocessing.image import ImageDataGenerator
    # datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True,
    #                              rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
    #                              horizontal_flip=True)
    # print("training CNN...")
    # history = model.fit_generator(datagen.flow(traindata,trainlabel,batch_size=50),steps_per_epoch=len(traindata)/50,
    #                               epochs=100, validation_data=(testdata, testlabel), validation_steps=len(testdata)/50)
    history = model.fit(traindata, trainlabel, epochs=100, batch_size=50, validation_data=(testdata, testlabel))    # , callbacks = [EarlyStopping(monitor='val_acc', patience=2)] - loss: 0.9583 - acc: 0.7375  - val_loss: 1.4746 - val_acc: 0.6685
    # model.save(filepath=r'D:\pythonWork\indoor Location\third-code\model_weights-v10.h5')
    # model.save_weights('model_weights-v15.h5')
    with open('logLab.txt','wb') as file_txt:
        pickle.dump(history.history, file_txt)
    plotModelHistory(history)

def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def main():
    print('generate dataset')
    originalCSI, label = rawCSI()   # 原始振幅数据集与标签(317, 3*30*1500)，(317, 2)
    originalData = np.array(originalCSI[:, 0:3*30*900], dtype='float')
    originalData = SimpleImputer(copy=False).fit_transform(originalData)

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler().fit(originalData)
    data_nomal = scaler.transform(originalData)

    data, phaseLabel =phaseMatrix() # 相位图像数据集与标签(317*30, 30, 30, 3)，(317*30, 1)
    data = np.array(data, dtype='float') / 255.0

    '原始振幅数据集分割'
    X = data_nomal
    Y = label
    trainRaw, testRaw, trainRawlabel, testRawlabel = train_test_split(X, Y, test_size=0.2, random_state=10)

    print('data saving_________________')
    np.save("47train",trainRaw)
    np.save("47test", testRaw)
    np.save("47trainLabel", trainRawlabel)
    np.save("47testLabel", testRawlabel)

    '相位图像数据集分割'
    # traindata, trainlabel = splitDataSet(data, Y, trainRawlabel)  #效果不理想
    # testdata, testlabel = splitDataSet(data, Y, testRawlabel)
    # print('train CNN model')
    # swapLabel = np.zeros((317 * 30, 317), dtype=int)
    # newlist = np.hstack((trainlabel,testlabel))  # 二值化标签
    # for i in range(317 * 30):
    #     swapLabel[i, newlist[i] ] = 1
    # trainlabel = swapLabel[0 : 253 * 30, :]
    # testlabel = swapLabel[253 * 30 : 317 * 30, :]
    # print(len(traindata), len(trainlabel), len(testdata), len(testlabel))

    numOfImage = 30
    labels = np.array(phaseLabel)
    lb = LabelBinarizer()
    phaseLabel = lb.fit_transform(labels)
    traindata, testdata, trainlabel, testlabel = train_test_split(data, phaseLabel, test_size=0.2, random_state=10)
    lbtestlabel = lb.inverse_transform(testlabel)
    print(len(traindata),len(testdata))

    # 'train and test CNN model'
    # time_start = time.time()
    # constructCNN(traindata, testdata, trainlabel, testlabel)
    # Retraining_time = time.time() - time_start
    # print('Retraining time is ', round(Retraining_time, 6), 'seconds')   # 790.002951 s

    'cluster the predicts of phase images for every position'
    from sklearn.cluster import KMeans
    Mymodel = load_model(r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/model-v10.h5')
    predict = Mymodel.predict_classes(testdata)
    cluster = []
    for i in range(317):
        list1 = np.where(lbtestlabel == (i+1))
        index = predict[list1]
        position = label[index]
        swap = []
        for i in range(len(position)):
            first = [position[i][0],position[i][1]]
            swap.append(first)
        if swap != []:
            kmeans = KMeans(n_clusters=1, random_state=0).fit(swap)
            cluster_center = kmeans.cluster_centers_
            cluster.append(cluster_center)
        else:
            cluster.append(np.reshape(label[i],(1,2)))
    predictOfPhaseImage= np.reshape(cluster,(317,2))

    'BLS classification regression'
    N1 = 30  # # of nodes belong to each window
    N2 = 5  # # of windows -------Feature mapping layer
    N3 = 100  # # of enhancement nodes -----Enhance layer
    L = 15  # # of incremental steps    14  retrain 2.2606016900807346 m
    M1 = 100  # # of adding enhance nodes
    M2 = 100  # # of adding feature mapping nodes
    M3 = 100  # # of adding enhance nodes
    s = 0.8  # shrink coefficient
    C = 2 ** -15  # Regularization coefficient
    print('-------------------BLS_BASE---------------------------')
    traindata = np.reshape(traindata, (len(traindata), 30 * 30 * 3))
    testdata = np.reshape(testdata, (len(testdata), 30 * 30 * 3))
    test_acc,test_time,train_acc_all,train_time,OutputOfTest = BLS_AddFeatureEnhanceNodes(traindata, trainlabel, testdata, testlabel, s, C, N1, N2, N3, L, M1,M2,M3)

    '找到每个类别的5个最大分类概率，进行初步期望回归，迭代次数为测试集长度'
    OutputOfTest = np.array(OutputOfTest)
    max_position = np.zeros((len(testdata), 5))
    max_position_weight = np.zeros((len(testdata), 5))
    list2 = []
    import heapq as hq
    for i in range(len(OutputOfTest)):
        maxpro = hq.nlargest(5, OutputOfTest[i])
        max_position_weight[i] = maxpro
        maxpsi = hq.nlargest(5, range(len(OutputOfTest[i])), OutputOfTest[i].__getitem__)
        max_position[i] = maxpsi
        result = 0
        for j in range(5):
            a = (np.multiply(max_position_weight[i][j], label[int(max_position[i][j])]))
            result += a
        list2.append(result)

    '找到测试集对应的坐标，每个采样点有[1, n]个初步期望回归，求平均，精度略有提升'
    list3 = []
    lbtestlabel = np.array(lbtestlabel)
    for i in range(317):
        index1 = np.where(lbtestlabel == (i+1))
        list4 = []
        for i in range(len(index1[0])):
            positionll = list2[index1[0][i]]
            list4.append(positionll)
            predictPositon = np.mean(list4, axis=0)
        list3.append(predictPositon)

    '将BLS计算的每个采样点最大分类概率作为权重w，与CNN聚类结果进行联合定位(1-w)，w的值很小'
    listPro = []
    for i in range(len(OutputOfTest)):
        probability = np.max(OutputOfTest[i])
        listPro.append(probability)
    list6 = []
    for i in range(317):
        index1 = np.where(lbtestlabel == (i + 1))
        list5 = []
        for i in range(len(index1[0])):
            position = listPro[index1[0][i]]
            list5.append(position)
            newPro = np.max(list5)
        list6.append(newPro)

    '--------estimation location----------'
    time_start = time.time()
    list7 = []
    for i in range(317):
        error = np.multiply(np.array(list3[i]), np.array(list6[i])) + np.multiply((1-np.array(list6[i])), predictOfPhaseImage[i])
        list7.append(error)
    Test_time = time.time() - time_start
    print('test time is ', round(Test_time, 6), 'seconds')  # Lab  0.001646 seconds
    accuracyLab(list7, label)   #   2.380084696194027 m
    # saveTestErrorMat(list7, label, 'ILCL-Lab-Error')

    'plot phase image and reload model'
    # reloadModel(r'C:\Users\dell\Desktop\47imaginary_part\saveModel\model-v11.h5', testdata, testlabel)
    # plotHeatMap()

def accuracyLab(label1, label2):
    error = np.asarray(label1 - label2)
    print(np.mean(np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)) * 50 / 100 , 'm')

def accuracyPre(predictions, labels):
    return  np.mean(np.sqrt(np.sum((predictions-labels)**2,1)))

def saveTestErrorMat(predictions , testlabel, fileName):
    error = np.asarray(predictions - testlabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 50 / 100
    savemat(fileName+'.mat', {'array': sample})

def plotHeatMap():
    import seaborn as sns
    data = loadmat(r'C:\Users\dell\Desktop\47imaginary_part\imaginary103.mat')
    test = data['myData']
    swap = test[:, :, 600:630]
    newd = np.rollaxis(swap, 0, 3)
    phase = abs(np.array(newd)[:,:,2])  # 0, 1, 2
    sns.set(style='whitegrid',color_codes=True,)
    sns.heatmap(phase, cbar=False)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 15,
             }
    plt.xlabel('Packets', font2)
    plt.ylabel('Channel index', font2)
    # plt.savefig('heatMapAntennas6NoBar.png', bbox_inches='tight', dpi=500)
    plt.show(dpi=500)

def reloadModel(filePath, testdata, testlabel):
    model = load_model(filePath)
    acc = model.evaluate(testdata, testlabel, batch_size=50)
    print('loss = '+ str(acc[0]))
    print('accuracy =' + str(acc[1]))

def plotModelHistory(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'])
    plt.show()

if __name__ == '__main__':
    main()
    pass