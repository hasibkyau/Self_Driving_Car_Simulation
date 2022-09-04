import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

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
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'],nBins)
    # print(hist)
    # print(bins)
    # there is no 0 and now we will create a center in bins
    center = (bins[:-1]+bins[1:])*0.5
    print(center)
    plt.bar(center,hist,width = 0.06)
    plt.show()
