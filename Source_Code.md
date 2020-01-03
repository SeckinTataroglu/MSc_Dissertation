```python

# APPENDIX 1: Global Maxima – Local Minima Source Code 

# Reading CSV

import pandas as pd
df = pd.read_csv(r'C:\Users\~\FileName.csv')
print(df)

# Cleaning The Data

df1 = df[(df["XR1A[1]"]!=0) & (df["XR1A[1]"]!=-999.25)]
df1.reset_index(drop=True, inplace=True)

print(df1)

# Finding The Max-Min and Linear Regression

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import scipy
from scipy import signal
figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
depth = 0
samplingFrequency = 20
shift = 20 *3.28084 # feet conversion
shiftMult = 0

# Slope Calculation Function
def slope(x1,x2,y1,y2):
    m = (y2-y1)/(x2-x1)
    return m

# Axis ticks and tick names
xAxis = np.linspace(1,512*20,512)
xAxis = np.arange(0, 512*samplingFrequency, 20)
yAxis = []
y_ticks = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]

# Singular Savitzky-Golay Filter
savgol = signal.savgol_filter(df1.loc[depth,"XR"+str(1)+"A[1]":"XR"+str(1)+"A[512]"], \
                              axis=0, window_length=41, polyorder=3)

plt.plot(xAxis, df1.loc[depth,"XR"+str(1)+"A[1]":"XR"+str(1)+"A[512]"], label="Raw Data")
plt.plot(xAxis, savgol, label="Filtered Data")
plt.legend()
plt.ylabel('Values')
plt.xlabel('Time (μs)')
plt.show()


# Max, Min1 and Min2 Identification
maxsVal = []
mins1Val = []
mins2Val = []

maxsLoc = []
mins1Loc = []
mins2Loc = []


figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')

# Raw Data - Location and Value Determination for Extreme Points
for i in range(8):
    
    y = df1.loc[depth,"XR"+str(i+1)+"A[1]":"XR"+str(i+1)+"A[512]"]
    
    # Max1
    max1 = np.max(y)
    max1Loc = np.where(y == np.amax(y))[0][0]
    maxsVal.append(max1-shift*shiftMult)
    maxsLoc.append(max1Loc*samplingFrequency)
    
    # Min1
    for j in range(510):
        m1 = slope(max1Loc-j, max1Loc-j-1, y[max1Loc-j], y[max1Loc-j-1])
        m2 = slope(max1Loc-j-1, max1Loc-j-2, y[max1Loc-j-1], y[max1Loc-j-2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min1 = y[max1Loc-j-1]
            min1Loc = max1Loc-j-1
            break
    mins1Val.append(min1-shift*shiftMult)
    mins1Loc.append(min1Loc*samplingFrequency)

    # Min2
    for k in range(510):
        m1 = slope(max1Loc+k, max1Loc+k+1, y[max1Loc+k], y[max1Loc+k+1])
        m2 = slope(max1Loc+k+1, max1Loc+k+2, y[max1Loc+k+1], y[max1Loc+k+2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min2 = y[max1Loc+k+1]
            min2Loc = k+1
            break 
    mins2Val.append(min2-shift*shiftMult)
    mins2Loc.append(min2Loc*samplingFrequency + max1Loc*samplingFrequency)
    
    plt.plot(xAxis, df1.loc[depth,"XR"+str(i+1)+"A[1]":"XR"+str(i+1)+"A[512]"]-shift*shiftMult)
    shiftMult = shiftMult + 1


    
# Filtered Data  - Location and Value Determination for Extreme Points
for i in range(8):
    
    filtered = signal.savgol_filter(df1.loc[depth,"XR"+str(i+1)+"A[1]":"XR"+str(i+1)+"A[512]"], \
                                    axis=0, window_length=51, polyorder=3)

    # Max1
    max1 = np.max(filtered)
    max1Loc = np.where(filtered == np.amax(filtered))[0][0]
    maxsVal.append(max1-shift*shiftMult)
    maxsLoc.append(max1Loc*samplingFrequency)

    # Min1
    for j in range(510):
        m1 = slope(max1Loc-j, max1Loc-j-1, filtered[max1Loc-j], filtered[max1Loc-j-1])
        m2 = slope(max1Loc-j-1, max1Loc-j-2, filtered[max1Loc-j-1], filtered[max1Loc-j-2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min1 = filtered[max1Loc-j-1]
            min1Loc = max1Loc-j-1
            break
    mins1Val.append(min1-shift*shiftMult)
    mins1Loc.append(int(min1Loc*samplingFrequency))
    
    for k in range(510):
        m1 = slope(max1Loc+k, max1Loc+k+1, filtered[max1Loc+k], filtered[max1Loc+k+1])
        m2 = slope(max1Loc+k+1, max1Loc+k+2, filtered[max1Loc+k+1], filtered[max1Loc+k+2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min2 = filtered[max1Loc+k+1]
            min2Loc = k+1
            break 
    mins2Val.append(min2-shift*shiftMult)
    mins2Loc.append(int(min2Loc*samplingFrequency + max1Loc*samplingFrequency))

    yAxis.append((filtered-shift*shiftMult)[0])
    plt.plot(xAxis, filtered-shift*shiftMult)
    shiftMult = shiftMult + 1

    
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Linear Regression starts based on the locations and values of the extreme points

# MAXS
Xmaxs = np.array(maxsLoc).reshape(-1,1) # values converts it into a numpy array
Ymaxs = np.array(maxsVal).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor_maxs = LinearRegression()  # create object for the class
linear_regressor_maxs.fit(Xmaxs, Ymaxs)  # perform linear regression
Y_pred_maxs = linear_regressor_maxs.predict(Xmaxs)  # make predictions

print("Maxs coef", linear_regressor_maxs.coef_, metrics.r2_score(Ymaxs, Y_pred_maxs))

plt.scatter(Xmaxs, Ymaxs)
plt.plot(Xmaxs, Y_pred_maxs, color='black')


# MINS1
Xmins1 = np.array(mins1Loc).reshape(-1,1) # values converts it into a numpy array
Ymins1 = np.array(mins1Val).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor_mins1 = LinearRegression()  # create object for the class
linear_regressor_mins1.fit(Xmins1, Ymins1)  # perform linear regression
Y_pred_mins1 = linear_regressor_mins1.predict(Xmins1)  # make predictions

print("Mins1 coef", linear_regressor_mins1.coef_, metrics.r2_score(Ymins1, Y_pred_mins1))

plt.scatter(Xmins1, Ymins1)
plt.plot(Xmins1, Y_pred_mins1, color='black')


# MINS2
Xmins2 = np.array(mins2Loc).reshape(-1,1) # values converts it into a numpy array
Ymins2 = np.array(mins2Val).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor_mins2 = LinearRegression()  # create object for the class
linear_regressor_mins2.fit(Xmins2, Ymins2)  # perform linear regression
Y_pred_mins2 = linear_regressor_mins2.predict(Xmins2)  # make predictions

print("Mins2 coef", linear_regressor_mins2.coef_, metrics.r2_score(Ymins2, Y_pred_mins2))

plt.scatter(Xmins2, Ymins2)
plt.plot(Xmins2, Y_pred_mins2, color='black')

plt.ylabel('Receivers')
plt.xlabel('Time (μs)')
plt.yticks(yAxis, y_ticks)
plt.title("X-Pole Sonic Logs at the depth of "+str(dfDiff.loc[depth,"DEPTH"]))
plt.show()


print("Maxs", maxsLoc)
print("Maxs", maxsVal)

print("mins1", mins1Loc)
print("mins1", mins1Val)

print("mins2", mins2Loc)
print("mins2", mins2Val)

Difference C - A

import numpy as np

columnNames = []

xAxis = np.linspace(1,512,512)
for r in range(8):
    for i in range(512):
        columnNames.append(str(r+1) + "_" + str(i+1))


        
dfReceiverA = df1.loc[:,df1.columns.str.contains('A')]
dfReceiverA.columns = columnNames

dfReceiverC = df1.loc[:,df1.columns.str.contains('C')]
dfReceiverC.columns = columnNames

dfDiff = dfReceiverC.subtract(dfReceiverA)
dfDiff.insert(0, column = "DEPTH", value = df1.iloc[:,0:1]) #Inserting df1's DEPTH values
print(dfDiff)

Diff Linear Regression C - A

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import scipy
from scipy import signal
figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
# depth = 193
depth = 0
samplingFrequency = 20
shift = 0.20 * 3.2808399 # feet conversion
shiftMult = 0

# Savgol Filter Parameters
winLen = 41
polyOrd = 3


# Slope Calculation Function
def slope(x1,x2,y1,y2):
    m = (y2-y1)/(x2-x1)
    return m


xAxis = np.linspace(1,512*20,512)
xAxis = np.arange(0, 512*samplingFrequency, 20)
yAxis = []
y_ticks = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]

savgol = signal.savgol_filter(dfDiff.loc[depth, str(1)+"_1": str(1)+"_512"], \
                              axis=0, window_length=winLen, polyorder=polyOrd)

plt.plot(xAxis, dfDiff.loc[depth, str(1)+"_1": str(1)+"_512"], label="Unfiltered")
plt.plot(xAxis, savgol, label="Filtered")
plt.legend(loc="best")
plt.xlabel("Time (Hz)")
plt.title("R1 Sonic Logs at the depth of "+str(dfDiff.loc[depth,"DEPTH"]))
plt.show()


# Max, Min1 and Min2 Identification
maxsVal = []
mins1Val = []
mins2Val = []

maxsLoc = []
mins1Loc = []
mins2Loc = []


figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')


# Raw Data

yFirstLine = dfDiff.loc[depth,"1_1":"1_512"]

maxN = np.max(yFirstLine)
minN = abs(np.min(yFirstLine))

for i in range(8):

    y = dfDiff.loc[depth,str(i+1)+"_1":str(i+1)+"_512"]
 
    # Max1
    if maxN >= minN:
        print("Global Maximum Worked")
        max1 = np.max(y)
        max1Loc = np.where(y == np.amax(y))[0][0]
    
    else:
        print("Global Minimum Worked")
        max1 = np.min(y)
        max1Loc = np.where(y == np.amin(y))[0][0]
    
    maxsVal.append(max1-shift*shiftMult)
    maxsLoc.append(max1Loc*samplingFrequency)
    
    # Min1
    for j in range(510):
        m1 = slope(max1Loc-j, max1Loc-j-1, y[max1Loc-j], y[max1Loc-j-1])
        m2 = slope(max1Loc-j-1, max1Loc-j-2, y[max1Loc-j-1], y[max1Loc-j-2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min1 = y[max1Loc-j-1]
            min1Loc = max1Loc-j-1
            break
    mins1Val.append(min1-shift*shiftMult)
    mins1Loc.append(min1Loc*samplingFrequency)
    
    
    
    # Min2
    for k in range(510):
        m1 = slope(max1Loc+k, max1Loc+k+1, y[max1Loc+k], y[max1Loc+k+1])
        m2 = slope(max1Loc+k+1, max1Loc+k+2, y[max1Loc+k+1], y[max1Loc+k+2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min2 = y[max1Loc+k+1]
            min2Loc = k+1
            break 
    mins2Val.append(min2-shift*shiftMult)
    mins2Loc.append(min2Loc*samplingFrequency + max1Loc*samplingFrequency)
    
    plt.plot(xAxis, dfDiff.loc[depth,str(i+1)+"_1":str(i+1)+"_512"]-shift*shiftMult)
    shiftMult = shiftMult + 1

    
# Filtered Data



filtered1stLine = signal.savgol_filter(dfDiff.loc[depth,"1_1":"1_512"], \
                                    axis=0, window_length=winLen, polyorder=polyOrd)

maxN = np.max(filtered1stLine)
minN = abs(np.min(filtered1stLine))


for i in range(8):
    filtered = signal.savgol_filter(dfDiff.loc[depth,str(i+1)+"_1":str(i+1)+"_512"], \
                                    axis=0, window_length=winLen, polyorder=polyOrd)
        
    # Max1
    if maxN >= minN:
        print("Global Maximum Worked")
        max1 = np.max(filtered)
        max1Loc = np.where(filtered == np.amax(filtered))[0][0]

    else:
        print("Global Minimum Worked")
        max1 = np.min(filtered)
        max1Loc = np.where(filtered == np.amin(filtered))[0][0]
        
    maxsVal.append(max1-shift*shiftMult)
    maxsLoc.append(int(max1Loc*samplingFrequency))

    # Min1
    for j in range(510):
        m1 = slope(max1Loc-j, max1Loc-j-1, filtered[max1Loc-j], filtered[max1Loc-j-1])
        m2 = slope(max1Loc-j-1, max1Loc-j-2, filtered[max1Loc-j-1], filtered[max1Loc-j-2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min1 = filtered[max1Loc-j-1]
            min1Loc = max1Loc-j-1
            break
    mins1Val.append(min1-shift*shiftMult)
    mins1Loc.append(int(min1Loc*samplingFrequency))

    # Min2
    for k in range(510):
        m1 = slope(max1Loc+k, max1Loc+k+1, filtered[max1Loc+k], filtered[max1Loc+k+1])
        m2 = slope(max1Loc+k+1, max1Loc+k+2, filtered[max1Loc+k+1], filtered[max1Loc+k+2])
        if (m1 > 0 and m2 < 0) or \
            (m1 < 0 and m2 > 0):
            min2 = filtered[max1Loc+k+1]
            min2Loc = k+1
            break 
    mins2Val.append(min2-shift*shiftMult)
    mins2Loc.append(int(min2Loc*samplingFrequency + max1Loc*samplingFrequency))

    yAxis.append((filtered-shift*shiftMult)[0])

    plt.plot(xAxis, filtered-shift*shiftMult)
    shiftMult = shiftMult + 1  
    
print("Global finished")   
    

    
import numpy as np
import matplotlib.pyplot as plt  # To visualize
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# MAXS
Xmaxs = np.array(maxsLoc).reshape(-1,1) # values converts it into a numpy array
Ymaxs = np.array(maxsVal).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor_maxs = LinearRegression()  # create object for the class
linear_regressor_maxs.fit(Xmaxs, Ymaxs)  # perform linear regression
Y_pred_maxs = linear_regressor_maxs.predict(Xmaxs) # make predictions
R2 = metrics.r2_score(Ymaxs, Y_pred_maxs)


# Finding the best fit

Xmaxs_Temp = Xmaxs
Ymaxs_Temp = Ymaxs
noisyReceivers = []
for i in range(8):
    itemX = Xmaxs[i]
    itemY = Ymaxs[i]
    
    Xmaxs_Temp = np.delete(Xmaxs_Temp, i) # X value is removed
    Xmaxs_Temp = np.array(Xmaxs_Temp).reshape(-1,1) # reshaping
    Ymaxs_Temp = np.delete(Ymaxs_Temp, i) # Y value is removed
    Ymaxs_Temp = np.array(Ymaxs_Temp).reshape(-1,1) # reshapinged
    
    linear_regressor_maxs.fit(Xmaxs_Temp, Ymaxs_Temp)  # perform linear regression
    Y_pred_maxs_Temp = linear_regressor_maxs.predict(Xmaxs_Temp)  # make New predictions
    R2_Temp = metrics.r2_score(Ymaxs_Temp, Y_pred_maxs_Temp) # New R2 value with 7 items
    
    if (R2_Temp-0.1) > R2 and R2_Temp >= 0.9:
        noisyReceivers.append(i)
    
    Xmaxs_Temp = np.insert(Xmaxs_Temp, i, itemX)
    Xmaxs_Temp = np.array(Xmaxs_Temp).reshape(-1,1) # reshaping
    Ymaxs_Temp = np.insert(Ymaxs_Temp, i, itemY)
    Ymaxs_Temp = np.array(Ymaxs_Temp).reshape(-1,1) # reshapinged

        
print("Noisy Receivers", noisyReceivers)    


# Cleaning Noisy Receiver
if len(noisyReceivers) > 0:
    for r in range(len(noisyReceivers)):

        # Maxs
        Xmaxs = np.delete(Xmaxs, noisyReceivers[r]) # Noisy receiver is removed
        Xmaxs = np.array(Xmaxs).reshape(-1,1) # reshaping
        Ymaxs = np.delete(Ymaxs, noisyReceivers[r]) # Noisy receiver is removed
        Ymaxs = np.array(Ymaxs).reshape(-1,1) # reshaping

        # Mins1
        Xmins1 = np.delete(mins1Loc, noisyReceivers[r]) # Noisy receiver is removed
        Xmins1 = np.array(Xmins1).reshape(-1,1) # values converts it into a numpy array
        Ymins1 = np.delete(mins1Val, noisyReceivers[r]) # Noisy receiver is removed
        Ymins1 = np.array(Ymins1).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column

        # Mins2
        Xmins2 = np.delete(mins2Loc, noisyReceivers[r]) # Noisy receiver is removed
        Xmins2 = np.array(Xmins2).reshape(-1,1) # values converts it into a numpy array
        Ymins2 = np.delete(mins2Val, noisyReceivers[r]) # Noisy receiver is removed
        Ymins2 = np.array(Ymins2).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column    

else:        
# # Re-Shaping Data
# #     Xmaxs = np.array(maxsLoc).reshape(-1,1) # values converts it into a numpy array
# #     Ymaxs = np.array(maxsVal).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
    
    Xmins1 = np.array(mins1Loc).reshape(-1,1) # values converts it into a numpy array
    Ymins1 = np.array(mins1Val).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column

    Xmins2 = np.array(mins2Loc).reshape(-1,1) # values converts it into a numpy array
    Ymins2 = np.array(mins2Val).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
    
    
    
# MAXS Re-Regression 
linear_regressor_maxs = LinearRegression()  # create object for the class
linear_regressor_maxs.fit(Xmaxs, Ymaxs)  # perform linear regression
Y_pred_maxs = linear_regressor_maxs.predict(Xmaxs) # make predictions
R2Maxs = metrics.r2_score(Ymaxs, Y_pred_maxs)


print("Maxs coef", linear_regressor_maxs.coef_, metrics.r2_score(Ymaxs, Y_pred_maxs))

plt.scatter(Xmaxs, Ymaxs)
plt.plot(Xmaxs, Y_pred_maxs, color='black')


# MINS1 Regression
linear_regressor_mins1 = LinearRegression()  # create object for the class
linear_regressor_mins1.fit(Xmins1, Ymins1)  # perform linear regression
Y_pred_mins1 = linear_regressor_mins1.predict(Xmins1)  # make predictions
R2Mins1 = metrics.r2_score(Ymins1, Y_pred_mins1)

print("Mins1 coef", linear_regressor_mins1.coef_, metrics.r2_score(Ymins1, Y_pred_mins1))

plt.scatter(Xmins1, Ymins1)
plt.plot(Xmins1, Y_pred_mins1, color='black')


# MINS2
linear_regressor_mins2 = LinearRegression()  # create object for the class
linear_regressor_mins2.fit(Xmins2, Ymins2)  # perform linear regression
Y_pred_mins2 = linear_regressor_mins2.predict(Xmins2)  # make predictions
R2Mins2 = metrics.r2_score(Ymins2, Y_pred_mins2)

print("Mins2 coef", linear_regressor_mins2.coef_, metrics.r2_score(Ymins2, Y_pred_mins2))

plt.scatter(Xmins2, Ymins2)
plt.plot(Xmins2, Y_pred_mins2, color='black')
   
averageSlow = (1/abs(linear_regressor_maxs.coef_)*R2Maxs + \
                1/abs(linear_regressor_mins1.coef_)*R2Mins1 + \
                1/abs(linear_regressor_mins2.coef_)*R2Mins2 ) / (R2Maxs+R2Mins1+R2Mins2)
print("averageSlow:", averageSlow)
print("Depth:", dfDiff.loc[depth,"DEPTH"])



plt.ylabel('Receivers')
plt.xlabel('Time (μs)')
plt.yticks(yAxis, y_ticks)
plt.title("X-Pole Sonic Logs at the depth of "+str(dfDiff.loc[depth,"DEPTH"]))
plt.show()

print("Maxs", maxsLoc)
print("Maxs", maxsVal)

print("mins1", mins1Loc)
print("mins1", mins1Val)

print("mins2", mins2Loc)
print("mins2", mins2Val)


Iterate All The Data

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
import pandas as pd  # To read data
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy
from scipy import signal

lastDepth = 4239
samplingFrequency = 20
shift = 0.2 * 3.2808399 # feet conversion
shiftMult = 0

# Savitzky–Golay filter Paramaters
winLen = 33
polyOrd = 3


def slope(x1,x2,y1,y2):
    m = (y2-y1)/(x2-x1)
    return m



# Preparation of Result Lists
dfDiffResults = pd.DataFrame(columns=["Index", "Depth", "Method",
                                      "Average Slowness",
                                      "Global Max Coef", "Local Min1 Coef", "Local Min2 Coef",
                                      "Global Max R2", "Local Min1 R2", "Local Min2 R2",
                                      
                                      "Global MaxLoc[1]","Global MaxLoc[2]","Global MaxLoc[3]", 
                                      "Global MaxLoc[4]","Global MaxLoc[5]","Global MaxLoc[6]",
                                      "Global MaxLoc[7]","Global MaxLoc[8]",
                                      
                                      "Local Min1Loc[1]","Local Min1Loc[2]","Local Min1Loc[3]",
                                      "Local Min1Loc[4]","Local Min1Loc[5]","Local Min1Loc[6]",
                                      "Local Min1Loc[7]","Local Min1Loc[8]",
                                      
                                      "Local Min2Loc[1]","Local Min2Loc[2]","Local Min2Loc[3]",
                                      "Local Min2Loc[4]","Local Min2Loc[5]","Local Min2Loc[6]",
                                      "Local Min2Loc[7]","Local Min2Loc[8]",
                                      
                                      "Global MaxValue[1]","Global MaxValue[2]","Global MaxValue[3]", 
                                      "Global MaxValue[4]","Global MaxValue[5]","Global MaxValue[6]",
                                      "Global MaxValue[7]","Global MaxValue[8]",
                                      
                                      "Local Min1Value[1]","Local Min1Value[2]","Local Min1Value[3]",
                                      "Local Min1Value[4]","Local Min1Value[5]","Local Min1Value[6]",
                                      "Local Min1Value[7]","Local Min1Value[8]",
                                      
                                      "Local Min2Value[1]","Local Min2Value[2]","Local Min2Value[3]",
                                      "Local Min2Value[4]","Local Min2Value[5]","Local Min2Value[6]",
                                      "Local Min2Value[7]","Local Min2Value[8]"
                                     
                                     ])


# xAxis = np.linspace(1,512,512)
xAxis = np.arange(0, 512*samplingFrequency, 20)

# Depth iteration
for depth in range(4240):
    
    print("Depth:", depth)

    try:

        # Max, Min1 and Min2 Identification
        max1 = None
        min1 = None
        min2 = None

        max1Loc = None
        min1Loc = None
        min2Loc = None

        maxsVal = []
        mins1Val = []
        mins2Val = []

        maxsLoc = []
        mins1Loc = []
        mins2Loc = []

        Xmaxs = None
        Xmins1 = None
        Xmins2 = None

        Ymaxs = None
        Ymins1 = None
        Ymins2 = None

        linear_regressor_maxs = None
        linear_regressor_mins1 = None
        linear_regressor_mins2 = None

        Y_pred_maxs = None
        Y_pred_mins1 = None
        Y_pred_mins2 = None


        shift = 0.2 * 3.2808399 # feet conversion
        shiftMult = 0


            # Raw Data

            yFirstLine = dfDiff.loc[depth,"1_1":"1_512"]

            maxN = np.max(yFirstLine)
            minN = abs(np.min(yFirstLine))

            for i in range(8):

                y = dfDiff.loc[depth,str(i+1)+"_1":str(i+1)+"_512"]

                # Max1
                if maxN >= minN:
                    method = "Global Maximum Worked"
                    max1 = np.max(y)
                    max1Loc = np.where(y == np.amax(y))[0][0]

                else:
                    method = "Global Minimum Worked"
                    max1 = np.min(y)
                    max1Loc = np.where(y == np.amin(y))[0][0]

                maxsVal.append(max1-shift*shiftMult)
                maxsLoc.append(max1Loc*samplingFrequency)

                # Min1
                for j in range(510):
                    m1 = slope(max1Loc-j, max1Loc-j-1, y[max1Loc-j], y[max1Loc-j-1])
                    m2 = slope(max1Loc-j-1, max1Loc-j-2, y[max1Loc-j-1], y[max1Loc-j-2])
                    if (m1 > 0 and m2 < 0) or \
                        (m1 < 0 and m2 > 0):
                        min1 = y[max1Loc-j-1]
                        min1Loc = max1Loc-j-1
                        break
                mins1Val.append(min1-shift*shiftMult)
                mins1Loc.append(min1Loc*samplingFrequency)

                # Min2
                for k in range(510):
                    m1 = slope(max1Loc+k, max1Loc+k+1, y[max1Loc+k], y[max1Loc+k+1])
                    m2 = slope(max1Loc+k+1, max1Loc+k+2, y[max1Loc+k+1], y[max1Loc+k+2])
                    if (m1 > 0 and m2 < 0) or \
                        (m1 < 0 and m2 > 0):
                        min2 = y[max1Loc+k+1]
                        min2Loc = k+1
                        break 
                mins2Val.append(min2-shift*shiftMult)
                mins2Loc.append(min2Loc*samplingFrequency + max1Loc*samplingFrequency)

                shiftMult = shiftMult + 1


        # Filtered Data

        filtered1stLine = signal.savgol_filter(dfDiff.loc[depth,"1_1":"1_512"], \
                                    axis=0, window_length=winLen, polyorder=polyOrd)

        maxN = np.max(filtered1stLine)
        minN = abs(np.min(filtered1stLine))


        for i in range(8):
            filtered = signal.savgol_filter(dfDiff.loc[depth,str(i+1)+"_1":str(i+1)+"_512"], \
                                            axis=0, window_length=winLen, polyorder=polyOrd)

            # Max1
            if maxN >= minN:
                method = "Global Maximum Worked"
                max1 = np.max(filtered)
                max1Loc = np.where(filtered == np.amax(filtered))[0][0]

            else:
                method = "Global Minimum Worked"
                max1 = np.min(filtered)
                max1Loc = np.where(filtered == np.amin(filtered))[0][0]

            maxsVal.append(max1-shift*shiftMult)
            maxsLoc.append(int(max1Loc*samplingFrequency))

            # Min1
            for j in range(510):
                m1 = slope(max1Loc-j, max1Loc-j-1, filtered[max1Loc-j], filtered[max1Loc-j-1])
                m2 = slope(max1Loc-j-1, max1Loc-j-2, filtered[max1Loc-j-1], filtered[max1Loc-j-2])
                if (m1 > 0 and m2 < 0) or \
                    (m1 < 0 and m2 > 0):
                    min1 = filtered[max1Loc-j-1]
                    min1Loc = max1Loc-j-1
                    break
            mins1Val.append(min1-shift*shiftMult)
            mins1Loc.append(int(min1Loc*samplingFrequency))

            # Min2
            for k in range(510):
                m1 = slope(max1Loc+k, max1Loc+k+1, filtered[max1Loc+k], filtered[max1Loc+k+1])
                m2 = slope(max1Loc+k+1, max1Loc+k+2, filtered[max1Loc+k+1], filtered[max1Loc+k+2])
                if (m1 > 0 and m2 < 0) or \
                    (m1 < 0 and m2 > 0):
                    min2 = filtered[max1Loc+k+1]
                    min2Loc = k+1
                    break 
            mins2Val.append(min2-shift*shiftMult)
            mins2Loc.append(int(min2Loc*samplingFrequency + max1Loc*samplingFrequency))

            shiftMult = shiftMult + 1

        # MAXS
        Xmaxs = np.array(maxsLoc).reshape(-1,1) # values converts it into a numpy array
        Ymaxs = np.array(maxsVal).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
        linear_regressor_maxs = LinearRegression()  # create object for the class
        linear_regressor_maxs.fit(Xmaxs, Ymaxs)  # perform linear regression
        Y_pred_maxs = linear_regressor_maxs.predict(Xmaxs) # make predictions
        R2 = metrics.r2_score(Ymaxs, Y_pred_maxs)


        # Finding the best fit

        Xmaxs_Temp = Xmaxs
        Ymaxs_Temp = Ymaxs
        noisyReceivers = []
        for i in range(8):
            itemX = Xmaxs[i]
            itemY = Ymaxs[i]

            Xmaxs_Temp = np.delete(Xmaxs_Temp, i) # X value is removed
            Xmaxs_Temp = np.array(Xmaxs_Temp).reshape(-1,1) # reshaping
            Ymaxs_Temp = np.delete(Ymaxs_Temp, i) # Y value is removed
            Ymaxs_Temp = np.array(Ymaxs_Temp).reshape(-1,1) # reshapinged

            linear_regressor_maxs.fit(Xmaxs_Temp, Ymaxs_Temp)  # perform linear regression
            Y_pred_maxs_Temp = linear_regressor_maxs.predict(Xmaxs_Temp)  # make New predictions
            R2_Temp = metrics.r2_score(Ymaxs_Temp, Y_pred_maxs_Temp) # New R2 value with 7 items

            if (R2_Temp-0.1) > R2 and R2_Temp >= 0.9:
                noisyReceivers.append(i)

            Xmaxs_Temp = np.insert(Xmaxs_Temp, i, itemX)
            Xmaxs_Temp = np.array(Xmaxs_Temp).reshape(-1,1) # reshaping
            Ymaxs_Temp = np.insert(Ymaxs_Temp, i, itemY)
            Ymaxs_Temp = np.array(Ymaxs_Temp).reshape(-1,1) # reshapinged


        print("Noisy Receivers", noisyReceivers)    


        # Cleaning Noisy Receiver
        if len(noisyReceivers) > 0:
            for r in range(len(noisyReceivers)):

                # Maxs
                Xmaxs = np.delete(Xmaxs, noisyReceivers[r]) # Noisy receiver is removed
                Xmaxs = np.array(Xmaxs).reshape(-1,1) # reshaping
                Ymaxs = np.delete(Ymaxs, noisyReceivers[r]) # Noisy receiver is removed
                Ymaxs = np.array(Ymaxs).reshape(-1,1) # reshaping

                # Mins1
                Xmins1 = np.delete(mins1Loc, noisyReceivers[r]) # Noisy receiver is removed
                Xmins1 = np.array(Xmins1).reshape(-1,1) # values converts it into a numpy array
                Ymins1 = np.delete(mins1Val, noisyReceivers[r]) # Noisy receiver is removed
                Ymins1 = np.array(Ymins1).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column

                # Mins2
                Xmins2 = np.delete(mins2Loc, noisyReceivers[r]) # Noisy receiver is removed
                Xmins2 = np.array(Xmins2).reshape(-1,1) # values converts it into a numpy array
                Ymins2 = np.delete(mins2Val, noisyReceivers[r]) # Noisy receiver is removed
                Ymins2 = np.array(Ymins2).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column    

        else:        
        # Re-Shaping Data
        
            Xmins1 = np.array(mins1Loc).reshape(-1,1) # values converts it into a numpy array
            Ymins1 = np.array(mins1Val).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column

            Xmins2 = np.array(mins2Loc).reshape(-1,1) # values converts it into a numpy array
            Ymins2 = np.array(mins2Val).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column



        # MAXS Re-Regression 
        linear_regressor_maxs = LinearRegression()  # create object for the class
        linear_regressor_maxs.fit(Xmaxs, Ymaxs)  # perform linear regression
        Y_pred_maxs = linear_regressor_maxs.predict(Xmaxs) # make predictions
        Y_maxs_R2 = metrics.r2_score(Ymaxs, Y_pred_maxs)

        # MINS1 Regression
        linear_regressor_mins1 = LinearRegression()  # create object for the class
        linear_regressor_mins1.fit(Xmins1, Ymins1)  # perform linear regression
        Y_pred_mins1 = linear_regressor_mins1.predict(Xmins1)  # make predictions
        Y_mins1_R2 = metrics.r2_score(Ymins1, Y_pred_mins1)
        
        # MINS2
        linear_regressor_mins2 = LinearRegression()  # create object for the class
        linear_regressor_mins2.fit(Xmins2, Ymins2)  # perform linear regression
        Y_pred_mins2 = linear_regressor_mins2.predict(Xmins2)  # make predictions
        Y_mins2_R2 = metrics.r2_score(Ymins2, Y_pred_mins2)

        averageSlow = (1/abs(linear_regressor_maxs.coef_)*Y_maxs_R2 + \
                        1/abs(linear_regressor_mins1.coef_)*Y_mins1_R2 + \
                        1/abs(linear_regressor_mins2.coef_)*Y_mins2_R2 ) / (Y_maxs_R2+Y_mins1_R2+Y_mins2_R2)
    

        # inserting values to results dataframe
        dfDiffResults.loc[depth,"Index"] = depth
        dfDiffResults.loc[depth,"Depth"] = dfDiff.loc[depth,"DEPTH"] 
        dfDiffResults.loc[depth,"Method"] = method

        dfDiffResults.loc[depth,"Average Slowness"] = averageSlow[0][0]
        dfDiffResults.loc[depth,"Global Max Coef"] = linear_regressor_maxs.coef_[0][0]
        dfDiffResults.loc[depth,"Local Min1 Coef"] = linear_regressor_mins1.coef_[0][0]
        dfDiffResults.loc[depth,"Local Min2 Coef"] = linear_regressor_mins2.coef_[0][0]

        dfDiffResults.loc[depth,"Global Max R2"] = Y_maxs_R2
        dfDiffResults.loc[depth,"Local Min1 R2"] = Y_mins1_R2
        dfDiffResults.loc[depth,"Local Min2 R2"] = Y_mins2_R2

        dfDiffResults.iloc[depth, 9:17] = maxsLoc
        dfDiffResults.iloc[depth, 17:25] = mins1Loc
        dfDiffResults.iloc[depth, 25:33] = mins2Loc

        dfDiffResults.iloc[depth,33:41] = maxsVal
        dfDiffResults.iloc[depth,41:49] = mins1Val
        dfDiffResults.iloc[depth,49:57] = mins2Val


    except:
        print(depth, "an error occured")
        print("")
        print("")

        
        dfDiffResults.loc[depth,"Index"] = depth
        dfDiffResults.loc[depth,"Depth"] = dfDiff.loc[depth,"DEPTH"]
        
dfDiffResults.to_csv(r'C:\Users\~\FileName.csv')
print(dfDiffResults)

















# APPENDIX 2: LSTM Pre-Processing Source Code
# Reading CSV - Data

import pandas as pd
df = pd.read_csv(r'C:\Users\~\FileName.csv')
print(df.shape)
print(df)

# Splitting The Data

dfX = df.iloc[:, 1:4097]
dfY = df.loc[:, "DTX"]

print(dfX.shape)
print(dfX)

print(dfY.shape)
print(dfY)


# DataFrames to Matrices

# importing numpy 
import numpy as np 
  
# output array 
randomLines = np.random.randint(low = 0, high = 25592, size = 10000) 
print(randomLines.shape)
print(len(randomLines))
print(randomLines)


# X Dataset to Matrix FILTERED
import numpy as np
import scipy
from scipy import signal

x_train = None
y_train = None


# Savgol Filter Parameters
winLen = 41
polyOrd = 3

# Scaler finds the absolute maximum in order to normalize the data
scaler = max(abs(dfX.loc[0, "1_1":"8_512"].min()), 
             abs(dfX.loc[0, "1_1":"8_512"].max())
            )


x_train = np.array([[signal.savgol_filter(dfX.loc[randomLines[0], "1_1":"1_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "2_1":"2_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "3_1":"3_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "4_1":"4_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "5_1":"5_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "6_1":"6_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "7_1":"7_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[0], "8_1":"8_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd)
                   ]])

y_train = np.array([dfY.loc[randomLines[0], ]])

for i in range(len(randomLines)-1):
    tempX = None
    tempY = None
    scaler = None
    scaler = max(abs(dfX.loc[randomLines[i+1], "1_1":"8_512"].min()), 
                 abs(dfX.loc[randomLines[i+1], "1_1":"8_512"].max())
                )
    
    tempX = np.array([[signal.savgol_filter(dfX.loc[randomLines[i+1], "1_1":"1_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "2_1":"2_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "3_1":"3_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "4_1":"4_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "5_1":"5_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "6_1":"6_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "7_1":"7_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd),
                     signal.savgol_filter(dfX.loc[randomLines[i+1], "8_1":"8_512"]/scaler, axis=0, window_length=winLen, polyorder=polyOrd)
                   ]])
    tempY = np.array([dfY.loc[randomLines[i+1], ]])
    
    x_train = np.insert(x_train, len(x_train), tempX, axis=0)
    y_train = np.insert(y_train, len(y_train), tempY, axis=0)
    
    
# Y Dataset to Matrix
y_data = y_train

y_data = np.reshape(y_data, (len(randomLines),1))
y_data = y_data.astype('float32')
print(type(y_data))
print(y_data.shape)

# Assigning the y_data to y_data_scaled in order to avoid to break the original arrays
y_train = y_data / 500
print(y_train)    
    

print("Merged")
print(type(x_train))
print(x_train.shape)

print(type(y_train))
print(y_train.shape)

# Numpy Arrays are saved in order to avoiding loading the raw files again
np.save(r"C:\Users\Seckin\OneDrive - University of Leicester\3. Dissertation\1. Python Codes\merged_x_data_arr", x_data)
np.save(r"C:\Users\Seckin\OneDrive - University of Leicester\3. Dissertation\1. Python Codes\merged_y_data_arr", y_data)

# Checking the range of the input and output arrays
print(np.min(x_train))
print(np.max(x_train))
print(np.min(y_train))
print(np.max(y_train))



# TENSORFLOW

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.optimizers import SGD, Adadelta
from keras.callbacks import EarlyStopping

# Building The Model
model = Sequential()

model.add(LSTM(512, input_shape=(8, 512), activation="relu", return_sequences=False))
model.add(Dense(512, activation="relu"))
model.add(Dense(100, activation="relu"))
model.add(Dense(50, activation="relu"))

model.add(Dense(1))
model.load_weights("Model_Master_Merged_19.h5")
model.compile(loss="mse", optimizer="adam") # , metrics=["mse", "accuracy"]

checkpointer = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=5, verbose=1, mode="auto")

history = model.fit(x_train, y_train, validation_split=0.5, callbacks=[checkpointer], epochs=10,  verbose=1) 

# SAVING THE MODEL

from keras.models import load_model
model.save("Model_Master_Merged_20.h5")

# PREDICTING

# LOADING THE MASTER MODEL
from keras.models import load_model
model = load_model("Model_Master_Merged_20.h5")

# importing numpy 
import numpy as np

# Loading pre-saved numpy arrays
x_data = np.load(r"C:\Users\Seckin\OneDrive - University of Leicester\3. Dissertation\1. Python Codes\00. merged_x_data_arr.npy")
y_data = np.load(r"C:\Users\Seckin\OneDrive - University of Leicester\3. Dissertation\1. Python Codes\00. merged_y_data_arr.npy")

# TESTING THE WHOLE DATA

realOutputList = []
predictionList = []

for i in range(len(randomLines)-1):
    x_new = np.reshape(x_train[i], (1, 8, 512))
    y_new = np.reshape(y_train[i], (1,1))
    
    # Model Prediction Based on The New Input
    predict = model.predict(x_new)
    realOutputList.append(y_train[i][0]*500)
    predictionList.append(predict[0][0]*500)

# COMPARISON OF REAL AND PREDICTED OUTPUTS
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

plt.plot(realOutputList)
plt.plot(predictionList)
plt.show()

# SCATTER PLOT IN ORDER TO COMPARE FITNESS
plt.scatter(realOutputList, predictionList)
plt.show()

# DRAWING LINEAR REGRESSION LINE OVER PREDICTION
# IN ORDER TO GET R2 VALUE

from sklearn.linear_model import LinearRegression
from sklearn import metrics

from matplotlib.pyplot import figure
figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')

# Output Regression
realValues = np.array(realOutputList).reshape(-1,1) # values converts it into a numpy array
predictedValues = np.array(predictionList).reshape(-1,1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor_output = LinearRegression()  # create object for the class
linear_regressor_output.fit(realValues, predictedValues)  # perform linear regression
Y_pred_output = linear_regressor_output.predict(realValues)  # make predictions

print("Maxs coef", linear_regressor_output.coef_, metrics.r2_score(realValues, predictedValues))

plt.scatter(realValues, predictedValues)
plt.plot(realValues, Y_pred_output, color='black')
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.legend(('R2={}'.format(round(metrics.r2_score(realValues, predictedValues),4)),), loc="upper left")
plt.title("Randomly Trained Data Results")

# TRAINING - TEST LOSS VISUALISATION
import matplotlib.pyplot as plt

figure(num=None, figsize=(9, 6), dpi=80, facecolor='w', edgecolor='k')

dfLoss = None

dfLoss = pd.DataFrame({'Loss':history.history["loss"]})
dfLoss.to_csv(r'C:\Users\~\FileName.csv')
plt.plot(history.history["loss"], label = "Training Loss")
plt.plot(history.history["val_loss"], label = "Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE Loss")
plt.show()





