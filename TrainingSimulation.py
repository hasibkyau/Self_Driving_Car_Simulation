from utlis import *

###STEP 01: Initialize Data
path = 'myData'
data = importDataInfo(path)

###Step 02: Balance Data
data = balanceData(data,display=True)

