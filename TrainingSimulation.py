from utlis import *
from sklearn.model_selection import train_test_split

### STEP 01: Initialize Data
path = 'myData'
data = importDataInfo(path)

### Step 02: Balance Data
data = balanceData(data,display=False)

### Step 03: Preparing for preprocessing
imagesPath, steerings = loadData(path,data)
# print(imagesPath[0], steerings[0])

### Step 04: Spliting off the data
xTrain, xVal, yTrain, yVal = train_test_split(imagesPath,steerings, test_size=0.2,random_state=5)
print('Total training images: ', len(xTrain))
print('Total validation images: ', len(xVal))

### Step 05: Data Augmentation


### Step 6: Preprocessing

