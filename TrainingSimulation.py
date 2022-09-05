from utlis import *

###STEP 01: Initialize Data
path = 'myData'
data = importDataInfo(path)

###Step 02: Balance Data
data = balanceData(data,display=False)

###Step 03: Preparing for preprocessing
imagesPath, steering = loadData(path,data)
print(imagesPath[0], steering[0])

