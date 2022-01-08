# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 15:09:38 2018

@author: HAN_RUIZHI yb77447@umac.mo OR  501248792@qq.com

This code is the first version of BLS Python. 
If you have any questions about the code or find any bugs
   or errors during use, please feel free to contact me.
If you have any questions about the original paper, 
   please contact the authors of related paper.
"""

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA 
import time
#from scipy import stats 
#import matplotlib.pyplot as plt

'''
#输出训练/测试准确率
'''
def show_accuracy(predictLabel,Label): 
    count = 0
    label_1 = np.zeros(Label.shape[0])
    predlabel = []
    label_1 = Label.argmax(axis = 1)
    predlabel = predictLabel.argmax(axis = 1)
    for j in list(range(Label.shape[0])):
        if label_1[j] == predlabel[j]:
            count += 1
    return (round(count/len(Label),5))
'''
激活函数
'''
def tansig(x):
    return (2/(1+np.exp(-2*x)))-1

def sigmoid(data):
    return 1.0/(1+np.exp(-data))
    
def linear(data):
    return data
    
def tanh(data):
    return (np.exp(data)-np.exp(-data))/(np.exp(data)+np.exp(-data))
    
def relu(data):
    return np.maximum(data,0)

def pinv(A,reg):
    return np.mat(reg*np.eye(A.shape[1])+A.T.dot(A)).I.dot(A.T)
'''
参数压缩
'''
def shrinkage(a,b):
    z = np.maximum(a - b, 0) - np.maximum( -a - b, 0)
    return z
'''
参数稀疏化
'''
def sparse_bls(A,b):
    lam = 0.001
    itrs = 50
    AA = A.T.dot(A)   
    m = A.shape[1]
    n = b.shape[1]
    x1 = np.zeros([m,n])
    wk = x1
    ok = x1
    uk = x1
    L1 = np.mat(AA + np.eye(m)).I
    L2 = (L1.dot(A.T)).dot(b)
    for i in range(itrs):
        ck = L2 + np.dot(L1,(ok - uk))
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok   
    return wk


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def BLS(train_x,train_y,test_x,test_y,s,c,N1,N2,N3):
#    u = 0
    L = 0
    train_x = preprocessing.scale(train_x,axis = 1)# ,with_mean = '0') #处理数据
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])
    Beta1OfEachWindow = []

    distOfMaxAndMin = []
    minOfEachWindow = []
    ymin = 0
    ymax = 1
    train_acc_all = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1; #生成每个窗口的权重系数，最后一行为偏差
#        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        #存储每个窗口的系数化权重
        Beta1OfEachWindow.append(betaOfEachWindow)
        #每个窗口的输出 T1
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
#        print('Feature nodes in window: max:',np.max(outputOfEachWindow),'min:',np.min(outputOfEachWindow))
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis=0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    #以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    #生成强化层权重
    if N1*N2>=N3:
        random.seed(67797325)
#        dim = N1*N2+1
#        temp_matric = stats.ortho_group(dim)
#        weightOfEnhanceLayer = temp_matric[:,0:N3]
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3))-1
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
#    print('Enhance nodes: max:',np.max(tempOfOutputOfEnhanceLayer),'min:',np.min(tempOfOutputOfEnhanceLayer))

    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)

    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = np.dot(pinvOfInput,train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    #训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc_all[0][0] = trainAcc
    train_time[0][0] = trainTime
    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1)#,with_mean = True,with_std = True) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()#测试计时开始
#  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] =(ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin
#  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
#  强化层输出
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    
#  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
#  最终测试输出   
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() #训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc * 100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime

    return test_acc,test_time,train_acc_all,train_time, OutputOfTest

#%%%%%%%%%%%%%%%%%%%%%%%%    
'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
l------步数
M------步长
'''


def BLS_AddEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M):
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0
    ymax = 1 #数据收缩上限
    ymin = 0 #数据收缩下限
    train_x = preprocessing.scale(train_x,axis = 1) #处理数据 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])
#    Beta1OfEachWindow = np.zeros([N2,train_x.shape[1]+1,N1])
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    Beta1OfEachWindow = []
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1; #生成每个窗口的权重系数，最后一行为偏差
#        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append( np.max(outputOfEachWindow,axis =0) - np.min(outputOfEachWindow,axis =0))
        minOfEachWindow.append(np.min(outputOfEachWindow,axis =0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    #以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    #生成强化层权重
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayer = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,c)
    OutputWeight = pinvOfInput.dot(train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    #训练输出
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime
    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()#测试计时开始
#  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (ymax - ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i]-ymin
#  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
#  强化层输出
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    
#  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
#  最终测试输出   
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() #训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增量增加强化节点
    '''
    parameterOfShrinkAdd = []
    for e in list(range(L)):
        time_start=time.time()
        if N1*N2>= M : 
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M)-1)
        else :
            random.seed(e)
            weightOfEnhanceLayerAdd = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
        
#        WeightOfEnhanceLayerAdd[e,:,:] = weightOfEnhanceLayerAdd
#        weightOfEnhanceLayerAdd = weightOfEnhanceLayer[:,N3+e*M:N3+(e+1)*M]
        tempOfOutputOfEnhanceLayerAdd = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayerAdd)
        parameterOfShrinkAdd.append(s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        D = pinvOfInput.dot(OutputOfEnhanceLayerAdd)
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeightEnd = pinvOfInput.dot(train_y)
        InputOfOutputLayer = tempOfLastLayerInput
        Training_time = time.time() - time_start
        train_time[0][e+1] = Training_time
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        
        #增量增加节点的 测试过程
        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e]);
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        TestingAcc = show_accuracy(OutputOfTest1,test_y)
        
        Test_time = time.time() - time_start
        test_time[0][e+1] = Test_time
        test_acc[0][e+1] = TestingAcc
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' );
        
    return test_acc,test_time,train_acc,train_time
'''
增加强化层节点版---BLS

参数列表：
s------收敛系数
c------正则化系数
N1-----映射层每个窗口内节点数
N2-----映射层窗口数
N3-----强化层节点数
L------步数

M1-----增加映射节点数
M2-----与增加映射节点对应的强化节点数
M3-----新增加的强化节点
'''
#%%%%%%%%%%%%%%%%
def BLS_AddFeatureEnhanceNodes(train_x,train_y,test_x,test_y,s,c,N1,N2,N3,L,M1,M2,M3):
    
    #生成映射层
    '''
    两个参数最重要，1）y;2)Beta1OfEachWindow
    '''
    u = 0
    ymax = 1
    ymin = 0
    train_x = preprocessing.scale(train_x,axis = 1) 
    FeatureOfInputDataWithBias = np.hstack([train_x, 0.1 * np.ones((train_x.shape[0],1))])
    OutputOfFeatureMappingLayer = np.zeros([train_x.shape[0],N2*N1])
#    Beta1OfEachWindow = np.zeros([N2,train_x.shape[1]+1,N1])###############################
#    Beta1OfEachWindow2 = np.zeros([L,train_x.shape[1]+1,M1])
    Beta1OfEachWindow = list()
    distOfMaxAndMin = []
    minOfEachWindow = []
    train_acc = np.zeros([1,L+1])
    test_acc = np.zeros([1,L+1])
    train_time = np.zeros([1,L+1])
    test_time = np.zeros([1,L+1])
    time_start=time.time()#计时开始
    for i in range(N2):
        random.seed(i+u)
        weightOfEachWindow = 2 * random.randn(train_x.shape[1]+1,N1)-1; #生成每个窗口的权重系数，最后一行为偏差
#        WeightOfEachWindow([],[],i) = weightOfEachWindow; #存储每个窗口的权重系数
        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler1 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(FeatureOfEachWindow)
        FeatureOfEachWindowAfterPreprocess = scaler1.transform(FeatureOfEachWindow)
        #通过稀疏化计算映射层每个窗口内的最终权重
        betaOfEachWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfEachWindow)
        outputOfEachWindow = np.dot(FeatureOfInputDataWithBias,betaOfEachWindow)
        distOfMaxAndMin.append(np.max(outputOfEachWindow,axis = 0) - np.min(outputOfEachWindow,axis = 0))
        minOfEachWindow.append(np.mean(outputOfEachWindow,axis = 0))
        outputOfEachWindow = (outputOfEachWindow-minOfEachWindow[i])/distOfMaxAndMin[i]
        OutputOfFeatureMappingLayer[:,N1*i:N1*(i+1)] = outputOfEachWindow
        del outputOfEachWindow 
        del FeatureOfEachWindow 
        del weightOfEachWindow 
        
    #生成强化层
    #以下为映射层输出加偏置（强化层输入）
    InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])
    #生成强化层权重
    if N1*N2>=N3:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        weightOfEnhanceLayer = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
    
    tempOfOutputOfEnhanceLayer = np.dot(InputOfEnhanceLayerWithBias,weightOfEnhanceLayer)
    parameterOfShrink = s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    #生成最终输入
    InputOfOutputLayerTrain = np.hstack([OutputOfFeatureMappingLayer,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayerTrain,c)
    OutputWeight =pinvOfInput.dot(train_y) #全局违逆
    time_end=time.time() #训练完成
    trainTime = time_end - time_start
    
    #训练输出
    OutputOfTrain = np.dot(InputOfOutputLayerTrain,OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    print('Training accurate is' ,trainAcc*100,'%')
    print('Training time is ',trainTime,'s')
    train_acc[0][0] = trainAcc
    train_time[0][0] = trainTime
    #测试过程
    test_x = preprocessing.scale(test_x,axis = 1) #处理数据 x = (x-mean(x))/std(x) x属于[-1，1]
    FeatureOfInputDataWithBiasTest = np.hstack([test_x, 0.1 * np.ones((test_x.shape[0],1))])
    OutputOfFeatureMappingLayerTest = np.zeros([test_x.shape[0],N2*N1])
    time_start=time.time()#测试计时开始
#  映射层
    for i in range(N2):
        outputOfEachWindowTest = np.dot(FeatureOfInputDataWithBiasTest,Beta1OfEachWindow[i])
        OutputOfFeatureMappingLayerTest[:,N1*i:N1*(i+1)] = (ymax-ymin)*(outputOfEachWindowTest-minOfEachWindow[i])/distOfMaxAndMin[i] - ymin
#  强化层
    InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest, 0.1 * np.ones((OutputOfFeatureMappingLayerTest.shape[0],1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)
#  强化层输出
    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    
#  最终层输入
    InputOfOutputLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,OutputOfEnhanceLayerTest])
#  最终测试输出   
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() #训练完成
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    print('Testing accurate is' ,testAcc*100,'%')
    print('Testing time is ',testTime,'s')
    test_acc[0][0] = testAcc
    test_time[0][0] = testTime
    '''
        增加Mapping 和 强化节点
    '''
    WeightOfNewFeature2 = list()
    WeightOfNewFeature3 = list()
    for e in list(range(L)):
        time_start = time.time()
        random.seed(e+N2+u)
        weightOfNewMapping = 2 * random.random([train_x.shape[1]+1,M1]) - 1
        NewMappingOutput = FeatureOfInputDataWithBias.dot(weightOfNewMapping)
#        FeatureOfEachWindow = np.dot(FeatureOfInputDataWithBias,weightOfEachWindow) #生成每个窗口的特征
        #压缩每个窗口特征到[-1，1]
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(NewMappingOutput)
        FeatureOfEachWindowAfterPreprocess = scaler2.transform(NewMappingOutput)
        betaOfNewWindow  =  sparse_bls(FeatureOfEachWindowAfterPreprocess,FeatureOfInputDataWithBias).T
        Beta1OfEachWindow.append(betaOfNewWindow)
   
        TempOfFeatureOutput = FeatureOfInputDataWithBias.dot(betaOfNewWindow)
        distOfMaxAndMin.append( np.max(TempOfFeatureOutput,axis = 0) - np.min(TempOfFeatureOutput,axis = 0))
        minOfEachWindow.append(np.mean(TempOfFeatureOutput,axis = 0))
        outputOfNewWindow = (TempOfFeatureOutput-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e]
        #新的映射层整体输出
        OutputOfFeatureMappingLayer = np.hstack([OutputOfFeatureMappingLayer,outputOfNewWindow])
        # 新增加映射窗口的输出带偏置
        NewInputOfEnhanceLayerWithBias = np.hstack([outputOfNewWindow, 0.1 * np.ones((outputOfNewWindow.shape[0],1))])
        #新映射窗口对应的强化层节点，M2列
        if M1 >= M2:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2])-1)
        else:
            random.seed(67797325)
            RelateEnhanceWeightOfNewFeatureNodes = LA.orth(2*random.random([M1+1,M2]).T-1).T  
        WeightOfNewFeature2.append(RelateEnhanceWeightOfNewFeatureNodes)
        
        tempOfNewFeatureEhanceNodes = NewInputOfEnhanceLayerWithBias.dot(RelateEnhanceWeightOfNewFeatureNodes)
        
        parameter1 = s/np.max(tempOfNewFeatureEhanceNodes)
        #与新增的Feature Mapping 节点对应的强化节点输出
        outputOfNewFeatureEhanceNodes = tansig(tempOfNewFeatureEhanceNodes * parameter1)

        if N2*N1+e*M1>=M3:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3) - 1)
        else:
            random.seed(67797325+e)
            weightOfNewEnhanceNodes = LA.orth(2 * random.randn(N2*N1+(e+1)*M1+1,M3).T-1).T
        WeightOfNewFeature3.append(weightOfNewEnhanceNodes)
        # 整体映射层输出带偏置
        InputOfEnhanceLayerWithBias = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((OutputOfFeatureMappingLayer.shape[0],1))])

        tempOfNewEnhanceNodes = InputOfEnhanceLayerWithBias.dot(weightOfNewEnhanceNodes)
        parameter2 = s/np.max(tempOfNewEnhanceNodes)
        OutputOfNewEnhanceNodes = tansig(tempOfNewEnhanceNodes * parameter2);
        OutputOfTotalNewAddNodes = np.hstack([outputOfNewWindow,outputOfNewFeatureEhanceNodes,OutputOfNewEnhanceNodes])
        tempOfInputOfLastLayes = np.hstack([InputOfOutputLayerTrain,OutputOfTotalNewAddNodes])
        D = pinvOfInput.dot(OutputOfTotalNewAddNodes)
        C = OutputOfTotalNewAddNodes - InputOfOutputLayerTrain.dot(D)
        
        if C.all() == 0:
            w = D.shape[1]
            B = (np.eye(w)- D.T.dot(D)).I.dot(D.T.dot(pinvOfInput))
        else:
            B = pinv(C,c)
        pinvOfInput = np.vstack([(pinvOfInput - D.dot(B)),B])
        OutputWeight = pinvOfInput.dot(train_y)        
        InputOfOutputLayerTrain = tempOfInputOfLastLayes
        
        time_end = time.time()
        Train_time = time_end - time_start
        train_time[0][e+1] = Train_time
        predictLabel = InputOfOutputLayerTrain.dot(OutputWeight)
        TrainingAccuracy = show_accuracy(predictLabel,train_y)
        train_acc[0][e+1] = TrainingAccuracy
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        
        # 测试过程
        #先生成新映射窗口输出
        time_start = time.time() 
        WeightOfNewMapping =  Beta1OfEachWindow[N2+e]

        outputOfNewWindowTest = FeatureOfInputDataWithBiasTest.dot(WeightOfNewMapping )
         #TT1
        outputOfNewWindowTest = (ymax - ymin)*(outputOfNewWindowTest-minOfEachWindow[N2+e])/distOfMaxAndMin[N2+e] - ymin
        ## 整体映射层输出
        OutputOfFeatureMappingLayerTest = np.hstack([OutputOfFeatureMappingLayerTest,outputOfNewWindowTest])
        # HH2
        InputOfEnhanceLayerWithBiasTest = np.hstack([OutputOfFeatureMappingLayerTest,0.1*np.ones([OutputOfFeatureMappingLayerTest.shape[0],1])])
        # hh2
        NewInputOfEnhanceLayerWithBiasTest = np.hstack([outputOfNewWindowTest,0.1*np.ones([outputOfNewWindowTest.shape[0],1])])

        weightOfRelateNewEnhanceNodes = WeightOfNewFeature2[e]
        #tt22
        OutputOfRelateEnhanceNodes = tansig(NewInputOfEnhanceLayerWithBiasTest.dot(weightOfRelateNewEnhanceNodes) * parameter1)
        #
        weightOfNewEnhanceNodes = WeightOfNewFeature3[e]
        # tt2
        OutputOfNewEnhanceNodes = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfNewEnhanceNodes)*parameter2)
        
        InputOfOutputLayerTest = np.hstack([InputOfOutputLayerTest,outputOfNewWindowTest,OutputOfRelateEnhanceNodes,OutputOfNewEnhanceNodes])
    
        predictLabel = InputOfOutputLayerTest.dot(OutputWeight)

        TestingAccuracy = show_accuracy(predictLabel,test_y)
        time_end = time.time()
        Testing_time= time_end - time_start
        test_time[0][e+1] = Testing_time;
        test_acc[0][e+1]=TestingAccuracy;
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' );


    return test_acc,test_time,train_acc,train_time, predictLabel
'''

'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def bls_train_input(train_x,train_y,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,l,m):

#%Incremental Learning Process of the proposed broad learning system: for
#%increment of input patterns
#%Input: 
#%---train_x,test_x : the training data and learning data in the begining of
#%the incremental learning
#%---train_y,test_y : the label
#%---train_yf,train_xf: the whold training samples of the learning system
#%---We: the randomly generated coefficients of feature nodes
#%---wh:the randomly generated coefficients of enhancement nodes
#%----s: the shrinkage parameter for enhancement nodes
#%----C: the regularization parameter for sparse regualarization
#%----N1: the number of feature nodes  per window
#%----N2: the number of windows of feature nodes
#%----N3: the number of enhancements nodes
#% ---m:number of added input patterns per increment step
#% ---l: steps of incremental learning
#
#%output:
#%---------Testing_time1:Accumulative Testing Times
#%---------Training_time1:Accumulative Training Time
    u = 0 #random seed
    ymin = 0
    ymax = 1 
    train_err = np.zeros([1,l+1])
    test_err = np.zeros([1,l+1])
    train_time = np.zeros([1,l+1])
    test_time = np.zeros([1,l+1])
    minOfEachWindow = []
    distMaxAndMin = []
    beta11 = list()
    Wh = list()
    '''
    feature nodes
    '''
    time_start = time.time()
    train_x = preprocessing.scale(train_x,axis = 1) 
    H1 = np.hstack([train_x, .1 * np.ones([train_x.shape[0],1])])
    y = np.zeros([train_x.shape[0],N2*N1]);
    for i in range(N2):
        random.seed(i+u)
        we= 2 * random.randn(train_x.shape[1]+1,N1)-1
        A1 = H1.dot(we)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler2.transform(A1)
        beta1 =  sparse_bls(A1,H1).T
        beta11.append(beta1)
        T1 = H1.dot(beta1)
        minOfEachWindow.append(T1.min(axis = 0))
        distMaxAndMin.append( T1.max(axis = 0) - T1.min(axis = 0))
        T1 = (T1 - minOfEachWindow[i])/distMaxAndMin[i]
        y[:,N1*i:N1*(i+1)] = T1
        
    '''
    enhancement nodes
    '''
    H2 = np.hstack([y,0.1 * np.ones([y.shape[0],1])])
    if N1*N2>=N3 :
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        wh = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T                 
    Wh.append(wh)
    T2 = H2.dot(wh)
    parameter = s/np.max(T2)
    T2 = tansig(T2 * parameter);
    T3 = np.hstack([y,T2])
    beta = pinv(T3,C)
    beta2 = beta.dot(train_y)
    Training_time = time.time() - time_start
    train_time[0][0] =Training_time;
    print('Training has been finished!');
    print('The Total Training Time is : ', Training_time, ' seconds' );
    xx = T3.dot(beta2)
    '''
    Training Accuracy
    '''
    TrainingAccuracy = show_accuracy(xx,train_y)
    print('Training Accuracy is : ', TrainingAccuracy * 100, ' %' )
    train_err[0][0] = TrainingAccuracy;

    '''
    Testing Process
    '''
    time.time()
    test_x = preprocessing.scale(test_x,axis = 1) 
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])
    yy1=np.zeros([test_x.shape[0],N2*N1]);
    for i in range(N2):
        beta1 = beta11[i]
        TT1 = HH1.dot(beta1)
        TT1 = (ymax - ymin)*(TT1 - minOfEachWindow[i])/distMaxAndMin[i] - ymin
        yy1[:,N1*i:N1*(i+1)]= TT1
    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])]); 
    TT2 = tansig(HH2.dot(wh) * parameter)
    TT3 = np.hstack([yy1,TT2])

    '''
    testing accuracy
    '''
    x = TT3.dot( beta2)
    TestingAccuracy = show_accuracy(x,test_y)
    Testing_time = time.time()- time_start
    test_time[0][0] = Testing_time
    test_err[0][0] = TestingAccuracy;
    print('Testing has been finished!');
    print('The Total Testing Time is : ', Testing_time, ' seconds' );
    print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' );
    
    '''
    incremental training steps
    '''
    for e in range(l):
        time_start = time.time()
        '''
   WARNING: If data comes from a single dataset, the following 'train_xx' and 'train_y1' should be reset!
        '''
        train_xx = preprocessing.scale(train_xf[(3000+(e)*m):(3000+(e+1)*m),:],axis = 1)
        train_y1 = train_yf[0:3000+(e+1)*m,:]

        Hx1 = np.hstack([train_xx, 0.1 * np.ones([train_xx.shape[0],1])])
        yx = np.zeros([train_xx.shape[0],N1*N2])
        for i in range(N2):
            beta1 = beta11[i]
            Tx1 = Hx1.dot(beta1)
            Tx1 = (ymax - ymin)*(Tx1 - minOfEachWindow[i])/distMaxAndMin[i] - ymin
            yx[:,N1*i:N1*(i+1)]= Tx1
                                       
        Hx2 = np.hstack([yx, 0.1 * np.ones([yx.shape[0],1])]);
        wh = Wh[0]
        t2 = tansig(Hx2.dot(wh) * parameter);
        t2 = np.hstack([yx, t2])
        betat = pinv(t2,C)
        beta = np.hstack([beta, betat])
        beta2 = beta.dot(train_y1)
        T3 = np.vstack([T3,t2])
        Training_time= time.time()- time_start
        train_time[0][e+1] = Training_time
        xx = T3.dot( beta2)
        TrainingAccuracy = show_accuracy(xx,train_y1)
        train_err[0][e+1] = TrainingAccuracy
        print('Training Accuracy is : ', TrainingAccuracy * 100, ' %' );
        '''
        incremental testing steps
        '''
        time_start = time.time()
        x = TT3.dot(beta2)
        TestingAccuracy = show_accuracy(x,test_y)
        Testing_time = time.time() - time_start
        test_time[0][e+1] = Testing_time
        test_err[0][e+1] = TestingAccuracy;
        print('Testing has been finished!')
        print('The Total Testing Time is : ', Testing_time, ' seconds' )
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' )
    return test_err,test_time,train_err,train_time

        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
def bls_train_inputenhance(train_x,train_y,train_xf,train_yf,test_x,test_y,s,C,N1,N2,N3,l,m,m2):
#
#%Incremental Learning Process of the proposed broad learning system: for
#%increment of input patterns
#%Input: 
#%---train_x,test_x : the training data and learning data in the begining of
#%the incremental learning
#%---train_y,test_y : the label
#%---train_yf,train_xf: the whold training samples of the learning system
#%---We: the randomly generated coefficients of feature nodes
#%---wh:the randomly generated coefficients of enhancement nodes
#%----s: the shrinkage parameter for enhancement nodes
#%----C: the regularization parameter for sparse regualarization
#%----N1: the number of feature nodes  per window
#%----N2: the number of windows of feature nodes
#%----N3: the number of enhancements nodes
#% ---m:number of added input patterns per incremental step
#%----m2:number of added enhancement nodes per incremental step
#% ----l: steps of incremental learning
#
#%output:
#%---------Testing_time1:Accumulative Testing Times
#%---------Training_time1:Accumulative Training Time
    u = 0
    ymax = 1
    ymin = 0
    train_err = np.zeros([1,l+1])
    test_err=np.zeros([1,l+1])
    train_time=np.zeros([1,l+1])
    test_time=np.zeros([1,l+1])
    l2 = []
    '''feature nodes'''
    time_start = time.time()
    train_x =  preprocessing.scale(train_x,axis = 1)
    H1 = np.hstack([train_x, 0.1 * np.ones([train_x.shape[0],1])])
    y = np.zeros([train_x.shape[0],N2*N1])
    beta11 = list()
    minOfEachWindow = []
    distMaxAndMin = []
    for i in range(N2):
        random.seed(i+u)
        we = 2 * random.randn(train_x.shape[1]+1,N1)-1
        A1 = H1.dot(we)
        scaler2 = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(A1)
        A1 = scaler2.transform(A1)
        beta1  =  sparse_bls(A1,H1).T
        beta11.append(beta1)
        T1 = H1.dot(beta1)
        minOfEachWindow.append(T1.min(axis = 0))
        distMaxAndMin.append(T1.max(axis = 0) - T1.min(axis = 0))
        T1 = (ymax - ymin)*(T1 - minOfEachWindow[i])/distMaxAndMin[i] - ymin
        y[:,N1*i:N1*(i+1)] = T1
    '''
    enhancement nodes%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    H2 = np.hstack([y, 0.1 * np.ones([y.shape[0],1])]);
    Wh = list()
    if N1*N2>=N3:
        random.seed(67797325)
        wh = LA.orth( 2 * random.randn(N2*N1+1,N3)-1)
    else:
        random.seed(67797325)
        wh = LA.orth( 2 * random.randn(N2*N1+1,N3).T-1).T
    Wh.append(wh)
    T2 = H2.dot(wh)
    l2.append( s/np.max(T2))
    T2 = tansig(T2 * l2[0])
    T3 = np.hstack([y, T2])
    beta = pinv(T3,C)
    beta2 = beta.dot(train_y)
    Training_time=time.time() - time_start
    train_time[0][0] =Training_time
    print('Training has been finished!')
    print('The Total Training Time is : ', Training_time, ' seconds' )
    '''
    %%%%%%%%%%%%%%%%%Training Accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    xx = T3.dot( beta2)

    TrainingAccuracy = show_accuracy(xx,train_y)
    print('Training Accuracy is : ', TrainingAccuracy * 100, ' %' )
    train_err[0][0] = TrainingAccuracy
    '''
    %%%%%%%%%%%%%%%%%%%%%%Testing Process%%%%%%%%%%%%%%%%%%%
    '''
    time_start = time.time()
    test_x = preprocessing.scale(test_x,axis = 1)
    HH1 = np.hstack([test_x, 0.1 * np.ones([test_x.shape[0],1])])

    yy1 = np.zeros([test_x.shape[0],N2*N1])

    for i in range(N2):
        beta1=beta11[i]
        TT1 = HH1.dot(beta1)
        TT1 = (ymax-ymin)*(TT1-minOfEachWindow[i])/distMaxAndMin[i] - ymin 
        yy1[:,N1*i:N1*(i+1)] = TT1
                             
    HH2 = np.hstack([yy1, 0.1 * np.ones([yy1.shape[0],1])])
    TT2 = tansig(HH2.dot(wh) * l2[0])
    TT3 = np.hstack([yy1, TT2])
    x = TT3.dot(beta2)
    TestingAccuracy = show_accuracy(x,test_y)
    '''        
    %%%%%%%%%%%%%%%%% testing accuracy%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    Testing_time = time.time()-time_start
    test_time[0][0] = Testing_time
    test_err[0][0] =TestingAccuracy
    print('Testing has been finished!');
    print('The Total Testing Time is : ', Testing_time, ' seconds' )
    print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' )
    '''
    %%%%%%%%%%%%%incremental training steps%%%%%%%%%%%%%%%%%%%
    '''

    for e in range(l):
        time_start = time.time()
        '''
   WARNING: If data comes from a single dataset, the following 'train_xx' and 'train_y1' should be reset!
        '''
        train_xx = preprocessing.scale(train_xf[(3000+(e)*m):(3000+(e+1)*m),:],axis = 1)
        train_y1 = train_yf[0:3000+(e+1)*m,:]
        Hx1 = np.hstack([train_xx, 0.1 * np.ones([train_xx.shape[0],1])])
        yx = np.zeros([train_xx.shape[0],N1*N2])
        for i in range(N2):
            beta1 = beta11[i]
            Tx1 = Hx1.dot(beta1)
            Tx1 = (ymax - ymin)*(Tx1-minOfEachWindow[i])/distMaxAndMin[i] - ymin  
            yx[:,N1*i:N1*(i+1)] = Tx1
        Hx2 = np.hstack([yx, 0.1 * np.ones([yx.shape[0],1])])
        tx22 = np.zeros([Hx2.shape[0],0])
        for o in range(e+1):
            wh = Wh[o]
            tx2 = Hx2.dot(wh)
            tx2 = tansig(tx2 * l2[o])
            tx22 = np.hstack([tx22, tx2])
            
        tx2x = np.hstack([yx,tx22])
        betat = pinv(tx2x,C)
        beta = np.hstack([beta, betat])
        T3 = np.vstack([T3,tx2x])
        y = np.vstack([y, yx])
        H2 = np.hstack([y, 0.1 * np.ones([y.shape[0],1])])
        if N1*N2>=m2:
#            random.seed(100+e)
            wh1 = LA.orth(2 * random.randn(N2*N1+1,m2)-1)
        else:
#            random.seed(100+e)
            wh1 = LA.orth(2 * random.randn(N2*N1+1,m2).T-1).T 
    
        Wh.append(wh1)
        t2 = H2.dot(wh1)
        l2.append( s/np.max(t2))
        t2 = tansig(t2 * l2[e+1])
        T3_temp = np.hstack([T3, t2])
        d = beta.dot(t2)
        c = t2 - T3.dot(d)
        if c.all()==0:
            w = d.shape[1]
            b= np.mat(np.eye(w)+d.T.dot(d)).I.dot(d.T.dot(beta))
        else:
            b = pinv(c,C)    
        beta = np.vstack([(beta-d.dot(b)),b])
        beta2 = beta.dot(train_y1)
        T3=T3_temp
        Training_time= time.time()-time_start
        train_time[0][e+1] = Training_time;
        xx = T3.dot(beta2)
        TrainingAccuracy = show_accuracy(xx,train_y1)
        train_err[0][e+1] = TrainingAccuracy
        print('Training Accuracy is : ', TrainingAccuracy * 100, ' %' )
#    %%%%%%%%%%%%%incremental testing steps%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        time_start = time.time()
        wh = Wh[e+1] 
        tt2 = tansig(HH2.dot(wh) * l2[e+1])
        TT3 = np.hstack([TT3, tt2])
        x = TT3.dot(beta2)
        TestingAccuracy = show_accuracy(x,test_y)
        Testing_time = time.time()-time_start
        test_time[0][e+1] = Testing_time
        test_err[0][e+1] = TestingAccuracy;
        print('Testing has been finished!');
        print('The Total Testing Time is : ',Testing_time, ' seconds' );
        print('Testing Accuracy is : ', TestingAccuracy * 100, ' %' );
    return test_err,test_time,train_err,train_time, x

