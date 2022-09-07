import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import cv2
import random

def getName(filePath):
    #getting the image name by spliting the filePath with slash ('\\') and selecting the last part ([-1])
    return filePath.split('\\')[-1]

def importDataInfo(path):
    #in myData there has the following coloumns
    coloums = ['Center','Left','Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'), names = coloums)
    # print(data.head())
    # print(data['Center'][0])
    # print(getName(data['Center'][0]))

    # we are applying the getName on the coloums of center camera and storing it on data
    data['Center'] = data['Center'].apply(getName)
    # print(data.head())
    print('Total imported image: ', data.shape[0])
    return data

def balanceData(data, display = True):
    nBins = 31
    samplesPerBin = 1000
    hist, bins = np.histogram(data['Steering'],nBins)
    # print(hist)
    # print(bins)
    # there is no 0 and now we will create a center in bins
    if display:
        center = (bins[:-1]+bins[1:])*0.5
        # print(center)
        plt.bar(center,hist,width = 0.06)
        # Cutting repeative value. So we will make a range of x axis from 1 to -1 and for y axis it should be 1000
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    # Removeoing redundant data
    removeIndexList = []
    for j in range(nBins):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j+1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeIndexList.extend((binDataList))

    print("Removed Imaged: ", len(removeIndexList))
    data.drop(data.index[removeIndexList], inplace = True)
    print("Remaining Images: ", len(data))

    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center,hist,width = 0.06)
        # Cutting repeative value. So we will make a range of x axis from 1 to -1 and for y axis it should be 1000
        plt.plot((-1,1),(samplesPerBin,samplesPerBin))
        plt.show()

    return data

def loadData(path, data):
    imagesPath = []
    steering = []

    for i in range (len(data)):
        indexedData = data.iloc[i]
        # print(indexedData)
        imagesPath.append(os.path.join(path,'IMG',indexedData[0]))
        # print(os.path.join(path,'IMG',indexedData[0]))
        steering.append(float(indexedData[3]))
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

# Augment image function
def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    ## PAN
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={'x': (-0.1, 0.1), 'y':(-0.1,0.1)})
        img = pan.augment_image(img)

    ## ZOOM
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)

    ## BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply(0.3,1.2)
        img = brightness.augment_image(img)

    ## FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img,1)
        steering = - steering

    return img, steering

# imgRe, st = augmentImage('test.jpg',0)
# plt.imshow(imgRe)
# plt.show()


## Croping image
def preProcessing(img):
    # croping the image
    img = img[60:134, :,:]
    # changing color
    img = cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3),0)
    img = img/255
    return img

imgRe = preProcessing(mpimg.imread('test.jpg'))
plt.imshow(imgRe)
plt.show()

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []

        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))

