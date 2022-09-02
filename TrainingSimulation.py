import pandas as pd
from sklearn.model_selection import train_test_split
from utlis import *

path = 'myData'
data = importDataInfo(path)
# data = pd.DataFrame(data)
# print(len(data))
# print(type(data))
# print(data)

data = balanceData(data, display=False)
# data = pd.DataFrame(data)
print(len(data))

imagesPath, steerings = loadData(path,data)

xTrain, xVal, yTrain, yVal = train_test_split(imagesPath, steerings, test_size=0.2,random_state=10)
print('Total Training Images: ',len(xTrain))
print('Total Validation Images: ',len(xVal))

model = createModel()
model.summary()