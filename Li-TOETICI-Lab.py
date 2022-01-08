from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, RMSprop
from keras.datasets import mnist
from keras.layers import LeakyReLU
from keras.layers import Deconv2D
from keras.preprocessing.image import image

import numpy as np
from PIL import Image
import argparse
import time
import math
from scipy.io import loadmat, savemat
import os
from sklearn.preprocessing import LabelBinarizer, Normalizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def getXlabel():
    xLabel = []
    for i in range(21):
        str = '%d' % (i + 1)
        xLabel.append(str)
    return xLabel

def getYlabel():
    yLabel = []
    for j in range(23):
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
    originalCSI=np.zeros((317, 3 * 30 * 1500), dtype=np.float)
    newName = []
    label = np.empty((0, 2), dtype=np.int)

    for i in range(21):
        for j in range(23):
            filePath = r"/Users/zhuxiaoqiang/Downloads/47SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
            name = xLabel[i] + yLabel[j]
            if (os.path.isfile(filePath)):
                c = loadmat(filePath)
                CSI = np.reshape(c['myData'], (1, 3 * 30 * 1500))
                originalCSI[count, :] = CSI[:, 3 * 30 * 960]
                newName.append(name)
                label = np.append(label, [[int(xLabel[i]), int(yLabel[j])]], axis=0)
                count += 1
    return originalCSI, label

def generateImage():
    midString = []
    xLabel = getXlabel()
    yLabel = getYlabel()
    datalist = []
    import cv2

    count = 0
    for i in range(21):
        for j in range(23):
            filePath = r"/Users/zhuxiaoqiang/Downloads/47SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                count +=1
                data = loadmat(filePath)
                test = data['myData']  # 3*30*1500  random 100

                for k in range(10):
                    swap = test[:, :, 128 * (k): 128 * (k + 1)]
                    terminalImage = np.reshape(swap, (1, 30 * 128 * 3))
                    terminalImage = terminalImage.reshape((30,128,3))
                    trans = image.array_to_img(terminalImage, scale=False)
                    resized = cv2.resize(src=image.img_to_array(trans), dsize=(256,256))
                    datalist.append(resized)

    Label = []
    numOfImage = 10
    count = 0
    for i in range(21):
        for j in range(23):
            filePath = r"/Users/zhuxiaoqiang/Downloads/47SwapData/coordinate" + xLabel[i] + yLabel[j] + ".mat"
            if (os.path.isfile(filePath)):
                count += 1
                Label.append(numOfImage*[count])
    Label = np.reshape(Label, (317*numOfImage, 1)).flatten()
    return datalist, Label

def Zscorenormalization(x):         #归一化
    x = ( x - np.mean(x)) / np.std(x)
    return x

def generator_model():
    chanDim = -1
    model = Sequential()

    # model.add(Dense(input_dim=100, output_dim=256*256*3))
    model.add(Dense(64 * 8 * 16 * 16))
    model.add(Reshape((16, 16, 512)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation('relu'))

    model.add(Deconv2D(256, (5, 5), padding='same', input_shape=(128,128,3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation('relu'))

    model.add(Deconv2D(128, (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation('relu'))

    model.add(Deconv2D(64, (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization(axis=chanDim))
    model.add(Activation('relu'))

    model.add(Deconv2D(3, (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('tanh'))
    return model

def discriminator_model():
    chanDim = -1
    model = Sequential()

    model.add(Conv2D(128, (5, 5), padding='same', input_shape=(256, 256, 3)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=chanDim))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(512, (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=chanDim))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(1024, (5, 5), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization(axis=chanDim))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(317, activation='softmax'))
    # model.add(Conv2D(1, (5, 5), padding='valid')) softmax
    # model.add(MaxPooling2D(pool_size=(1, 1)))
    return model

def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image

def saveTestErrorMat(predictions , testlabel, fileName):
    error = np.asarray(predictions - testlabel)
    sample = np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2) * 50 / 100
    savemat(fileName+'.mat', {'array': sample})

def accuracyLab(label1, label2):
    error = np.asarray(label1 - label2)
    print(np.mean(np.sqrt(error[:, 0] ** 2 + error[:, 1] ** 2)) * 50 / 100 , 'm')


def train(BATCH_SIZE):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    CSIdataset, label = rawCSI()
    print('generate dataset')
    originalCSI, labelplus = generateImage()    # 原始振幅数据集与标签(317*10, 256, 256, 3)，(317*10, 1)
    data = (np.array(originalCSI, dtype='float'))
    labels = np.array(labelplus)
    lb = LabelBinarizer()
    phaseLabel = lb.fit_transform(labels)

    traindata, testdata, trainlabel, testlabel = train_test_split(data, phaseLabel, test_size=0.2, random_state=5)
    lbtestlabel = lb.inverse_transform(testlabel)

    from keras.callbacks import Callback
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = RMSprop(lr=0.01, rho=0.9, epsilon=1e-05)
    g_optim = RMSprop(lr=0.01, rho=0.9, epsilon=1e-05)
    g.compile(loss='binary_crossentropy', optimizer="RMSprop")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)

    kerasCallback = Callback()
    # history = d.fit(traindata, trainlabel, epochs=20, batch_size=100, validation_data=(testdata, testlabel), callbacks=[kerasCallback])
    # d.save(filepath=r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/model-TOETICI-Lab-v1.h5')

    Mymodel = load_model(filepath=r'/Users/zhuxiaoqiang/Desktop/IEEE Trans/Third paper/代码/third-code/CompareExperiments/model-TOETICI-Lab-v1.h5')

    'estimate location'
    time_start = time.time()
    predict = Mymodel.predict_classes(testdata)
    from sklearn.cluster import KMeans
    cluster = []
    for i in range(317):
        list1 = np.where(lbtestlabel == (i + 1))
        index = predict[list1]
        position = label[index]
        swap = []
        for i in range(len(position)):
            first = [position[i][0], position[i][1]]
            swap.append(first)
        if swap != []:
            kmeans = KMeans(n_clusters=1, random_state=0).fit(swap)
            cluster_center = kmeans.cluster_centers_
            cluster.append(cluster_center)
        else:
            cluster.append(np.reshape(label[i], (1, 2)))
    predictOfPhaseImage = np.reshape(cluster, (317, 2))
    Test_time = time.time() - time_start
    print('test time is ', round(Test_time, 6), 'seconds')  # Lab  51.268913 seconds
    accuracyLab(predictOfPhaseImage, label)
    # saveTestErrorMat(predictOfPhaseImage, label, 'Li-TOETICI-Lab-Error')

def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    # if args.mode == "train":
    #     train(BATCH_SIZE=args.batch_size)
    # elif args.mode == "generate":
    #     generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    train(args.batch_size)

    # 5.739488434997771 m   random 5