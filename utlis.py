import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.utils import shuffle

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
