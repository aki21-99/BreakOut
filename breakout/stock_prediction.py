# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 22:22:37 2018

@author: Aditya
"""


#loading libraries
import os
os.chdir("D:\\Projects\\Stock22k\\one_minute_data\\2018")
import pandas as pd
import matplotlib.pyplot as plt 


#=====================Data collection=================================

data = pd.read_csv('ACC.txt', sep=",", header=None)
data.columns = ["Stock_name", "Date", "Time", "Open","High","Low","Close","Volume"]


#======================Data processing======================

#missing data
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("Missing Values in data \n\n",missing_data)


# replacing null values to 0
data = data.dropna(0)
data = data.reset_index(drop = True)

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print("Missing Values in data \n\n",missing_data)



#checking for outliers in scatterplot
print("Scatterplot")
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(data.index ,data['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing price')
plt.show()





#=====================Calculating Indicators=================================
def MA(df, n):  
    MA = pd.Series(pd.rolling_mean(df['Close'], n)) 
    return MA




#Exponential Moving Average  
def EMA(df, n):  
    EMA = pd.Series(pd.ewma(df['Close'], span = n, min_periods = n - 1))
    return EMA



#Momentum  
def MOM(df, n):  
    MOM = pd.Series(df['Close'].diff(n))
    return MOM



#Rate of Change  
def ROC(df, n):  
    M = df['Close'].diff(n - 1)  
    N = df['Close'].shift(n - 1)  
    ROC = pd.Series(M / N)  
    return ROC



#Average True Range  
def ATR(df, n):  
    i = 0  
    TR_l = [0]  
    while i < df.index[-1]:  
        TR = max(df.get_value(i + 1, 'High'), df.get_value(i, 'Close')) - min(df.get_value(i + 1, 'Low'), df.get_value(i, 'Close'))  
        TR_l.append(TR)  
        i = i + 1  
    TR_s = pd.Series(TR_l)  
    ATR = pd.Series(pd.ewma(TR_s, span = n, min_periods = n))
    return ATR



#Bollinger Bands  
def BBANDS(df, n):  
    MA = pd.Series(pd.rolling_mean(df['Close'], n))  
    MSD = pd.Series(pd.rolling_std(df['Close'], n))  
    b1 = 4 * MSD / MA  
    B1 = pd.Series(b1)  
    b2 = (df['Close'] - MA + 2 * MSD) / (4 * MSD)  
    B2 = pd.Series(b2)  
    return B1,B2



#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['High'])  
    R2 = pd.Series(PP + df['High'] - df['Low'])  
    S2 = pd.Series(PP - df['High'] + df['Low'])  
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
    psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    PSR = pd.DataFrame(psr)   
    return PSR



#Stochastic oscillator %K  
def STOK(df):  
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']))  
    return SOk



####
def VWAP(df):
    q = df.Volume.values
    p = df.Close.values
    VWAP=(p * q).cumsum() / q.cumsum()
    return VWAP




n = 20

data["MA"] = MA(data,n)
data["EMA"] = EMA(data,n)
data["MOM"] = MOM(data,n)
data["ROC"] = ROC(data,n)
data["ATR"] = ATR(data,n)
#data["B1","B2"] = BBANDS(data,20)
#data["PPSR"] = PPSR(data)
data["STOK"] = STOK(data)
data["VWAP"] = VWAP(data)
data["C-O"] = data["Close"] - data["Open"]
data["C-H"] = data["Close"] - data["High"]
#data.to_csv("ACC_2018_5mins.csv",index = False)








#=====================Labelling data=================================

window_size = 5


data['5_min'] = 0
for i in range(0,len(data)):
    if i+window_size <= len(data):
        values = data['Close'][i:i+window_size]
        start_value = data["Close"][i] 
        max_value = max(values)
        min_value = min(values)
        if max_value > (start_value + (0.0075 * start_value)):
            data['5_min'][i] = "breakout"
        elif min_value < (start_value - (0.0075 * start_value)):
            data['5_min'][i] = "breakdown"
        else:
            data['5_min'][i] = "normal"    
            
#
#if window_size = 5:    
#    data['5_min'] = 0
#    for i in range(0,len(data)):
#        if i+window_size <= len(data):
#            values = data['Close'][i:i+window_size]
#            start_value = data["Close"][i] 
#            max_value = max(values)
#            min_value = min(values)
#            if max_value > (start_value + (0.0075 * start_value)):
#                data['5_min'][i] = "breakout"
#            elif min_value < (start_value - (0.0075 * start_value)):
#                data['5_min'][i] = "breakdown"
#            else:
#                data['5_min'][i] = "normal"    
#elif window_size = 15:          
#    data['15_min'] = 0
#    for i in range(0,len(data)):
#        if i+window_size <= len(data):
#            values = data['Close'][i:i+window_size]
#            start_value = data["Close"][i] 
#            max_value = max(values)
#            min_value = min(values)
#            if max_value > (start_value + (0.015 * start_value)):
#                data['15_min'][i] = "breakout"
#            elif min_value < (start_value - (0.015 * start_value)):
#                data['15_min'][i] = "breakdown"
#            else:
#                data['15_min'][i] = "normal"   
#else:     
#    data['30_min'] = 0
#    for i in range(0,len(data)):
#        if i+window_size <= len(data):
#            values = data['Close'][i:i+window_size]
#            start_value = data["Close"][i] 
#            max_value = max(values)
#            min_value = min(values)
#            if max_value > (start_value + (0.02 * start_value)):
#              data['30_min'][i] = "breakout"
#            elif min_value < (start_value - (0.02 * start_value)):
#                data['30_min'][i] = "breakdown"  
#            else:
#             data['30_min'][i] = "normal"        
#         


#Looking the  features
dataset = data[n:len(data)-(window_size -1)]
dataset = dataset.reset_index(drop = True)
dataset = dataset.fillna(0)

cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA', 'EMA', 'MOM', 'ROC','ATR', 'STOK', 'VWAP','C-O', 'C-H']
columns = dataset.loc[:,cols].columns

breakdown = dataset['5_min'] == "breakdown"
breakout = dataset['5_min'] == "breakout"
normals = dataset["5_min"] == "normal"

import matplotlib.gridspec as gridspec
import seaborn as sns
grid = gridspec.GridSpec(14, 2)
plt.figure(figsize=(15,20*4))

for m, col in enumerate(dataset[columns]):
    ax = plt.subplot(grid[m])
    sns.distplot(dataset[col][breakout], bins = 50, color='g') #Will receive the "semi-salmon" violin
    sns.distplot(dataset[col][normals], bins = 50, color='r') #Will receive the "ocean" color
    sns.distplot(dataset[col][breakdown], bins = 50, color='b')
    ax.set_ylabel('Density')
    ax.set_title(str(col))
    ax.set_xlabel('')
plt.show()


from sklearn import preprocessing

import numpy as np
# Feature Importance
from sklearn import datasets
from sklearn import metrics
from xgboost import XGBClassifier



# separate the data from the target attributes
X = dataset.loc[:,cols]
y = dataset['5_min']

# normalize the data attributes
X = preprocessing.normalize(X)



# fit model no training data
model = XGBClassifier()
model.fit(X[:,5:], y)
# feature importance
print(model.feature_importances_)
# plot
print("\n Important Indicators")
print(cols[5:])
plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
plt.show()



    

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




#=========================splitting data into train and test data===================
seed = 7
test_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)



##===================== Modelling =================================
# fit model on training data
model = XGBClassifier()


model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#print(y_test, pd.Series(y_pred))
compare_table = pd.DataFrame(dict(y_test = y_test, y_pred = y_pred)).reset_index(drop = True)


from collections import Counter
Counter(compare_table['y_test'])
Counter(compare_table['y_pred'])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       


import sklearn.metrics as met
cm = met.confusion_matrix(y_test, y_pred)



ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(["breakdown",'Breakout', 'Normal']); ax.yaxis.set_ticklabels(['breakdown','Breakout', 'Normal']);


print ('\nClassification Report:\n-----------------------------')
print (met.classification_report(y_test, y_pred))


