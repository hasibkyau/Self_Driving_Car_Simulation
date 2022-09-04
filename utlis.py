import pandas as pd
import numpy as np
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