import pandas as pd
import numpy as np
import os

def importDataInfo(path):
    #in myData there has the following coloumns
    coloums = ['Center','Left','Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path,'driving_log.csv'), names = coloums)
    print(data.head())