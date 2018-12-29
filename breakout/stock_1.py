#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 00:24:11 2018

@author: Aditya Lashkare
"""

#loading libraries
import os
os.chdir("/home/admin/stock_prediction/Data") #set directory
import pandas as pd
import matplotlib.pyplot as plt 
pd.options.mode.chained_assignment = None
import glob


#=====================Data collection=================================
files = glob.glob("*csv")

dfs = []
for file in files:
    df = pd.read_csv(file, sep = ",", header = None)
    dfs.append(df)
   
data = pd.concat(dfs)
    
#data = pd.read_csv('all_stocks.csv', sep=",", header=None)
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


n = 20
#===============Technical Indicators
import talib

data["MA"] = talib.MA(data.Close,n )/data.Close
data["EMA"] = talib.EMA(data.Close,n)/data.Close
data["MOM"] = talib.MOM(data.Close,n)
data["ROC"] = talib.ROC(data.Close,n)
data["ATR"] = talib.ATR(data.High,data.Low,data.Close)
data["BBANDS"],data["BBANDS1"],data["BBANDS2"] = talib.BBANDS(data.Close)
data["BBANDS"],data["BBANDS1"],data["BBANDS2"] =data["BBANDS"]/data.Close,data["BBANDS1"]/data.Close,data["BBANDS2"]/data.Close


#-----------momentum indicators
#ADX - Average Directional Movement Index
data["ADX"] = talib.ADX(data.High, data.Low, data.Close, n)


#ADXR - Average Directional Movement Index Rating

data["ADXR"] = talib.ADXR(data.High, data.Low, data.Close, 14)

#APO - Absolute Price Oscillator

data["APO"] = talib.APO(data.Close, fastperiod=12, slowperiod=26, matype=0)

#Learn more about the Absolute Price Oscillator at tadoc.org.
#AROON - Aroon

data["aroondown"], data["aroonup"] = talib.AROON(data.High, data.Low,14)

#Learn more about the Aroon at tadoc.org.
#AROONOSC - Aroon Oscillator

data["AROONOSC"] = talib.AROONOSC(data.High, data.Low, 14)

#Learn more about the Aroon Oscillator at tadoc.org.
#BOP - Balance Of Power

data["BOP"] = talib.BOP(data.Open, data.High, data.Low, data.Close)

#Learn more about the Balance Of Power at tadoc.org.
#CCI - Commodity Channel Index

data["CCI"] = talib.CCI(data.High, data.Low, data.Close,14)

#Learn more about the Commodity Channel Index at tadoc.org.
#CMO - Chande Momentum Oscillator

#NOTE: The CMO function has an unstable period.

data["CMO"] = talib.CMO(data.Close, n)

#Learn more about the Chande Momentum Oscillator at tadoc.org.
#DX - Directional Movement Index

#NOTE: The DX function has an unstable period.

data["DX"] = talib.DX(data.High, data.Low, data.Close,n)


#Learn more about the Directional Movement Index at tadoc.org.
#MACD - Moving Average Convergence/Divergence

data["macd"], data["macdsignal"], data["macdhist"] = talib.MACD(data.Close, fastperiod=12, slowperiod=26, signalperiod=9)

#MFI - Money Flow Index

#NOTE: The MFI function has an unstable period.
#
data["MFI"] = talib.MFI(data.High, data.Low, data.Close, data.Volume, n)

#Learn more about the Money Flow Index at tadoc.org.
#MINUS_DI - Minus Directional Indicator

#NOTE: The MINUS_DI function has an unstable period.

data["MINUS_DI"] = talib.MINUS_DI(data.High, data.Low, data.Close, n)

#Learn more about the Minus Directional Indicator at tadoc.org.
#MINUS_DM - Minus Directional Movement

#NOTE: The MINUS_DM function has an unstable period.

data["MINUS_DI"] = talib.MINUS_DM(data.High, data.Low, n)

#Learn more about the Momentum at tadoc.org.
#PLUS_DI - Plus Directional Indicator

#NOTE: The PLUS_DI function has an unstable period.

data["PLUS_DI"] = talib.PLUS_DI(data.High, data.Low, data.Close, 14)

#Learn more about the Plus Directional Indicator at tadoc.org.
#PLUS_DM - Plus Directional Movement

#N1OTE: The PLUS_DM function has an unstable period.

data["PLUS_DM"] = talib.PLUS_DM(data.High, data.Low, 14)

#Learn more about the Plus Directional Movement at tadoc.org.
#PPO - Percentage Price Oscillator

data["PPO"] = talib.PPO(data.Close, fastperiod=12, slowperiod=26, matype=0)

#Learn more about the Percentage Price Oscillator at tadoc.org.
#ROC - Rate of change : ((price/prevPrice)-1)*100

data["ROC"] = talib.ROC(data.Close, timeperiod=10)

#Learn more about the Rate of change : ((price/prevPrice)-1)*100 at tadoc.org.
#ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice

data["ROCP"] = talib.ROCP(data.Close, timeperiod=10)

#Learn more about the Rate of change Percentage: (price-prevPrice)/prevPrice at tadoc.org.
#ROCR - Rate of change ratio: (price/prevPrice)

data["ROCR"] = talib.ROCR(data.Close, timeperiod=10)

#Learn more about the Rate of change ratio: (price/prevPrice) at tadoc.org.
#ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100

data["ROCR100"] = talib.ROCR100(data.Close, timeperiod=10)

#Learn more about the Rate of change ratio 100 scale: (price/prevPrice)*100 at tadoc.org.
#RSI - Relative Strength Index

#NOTE: The RSI function has an unstable period.

data["RSI"] = talib.RSI(data.Close, timeperiod=14)

#Learn more about the Relative Strength Index at tadoc.org.
#STOCH - Stochastic

data["slowk"], data["slowd"] = talib.STOCH(data.High, data.Low, data.Close, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

#Learn more about the Stochastic at tadoc.org.
#STOCHF - Stochastic Fast

data["fastk"], data["fastd"] = talib.STOCHF(data.High, data.Low, data.Close, fastk_period=5, fastd_period=3, fastd_matype=0)

#Learn more about the Stochastic Fast at tadoc.org.

#Learn more about the Stochastic Relative Strength Index at tadoc.org.
#TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA

data["TRIX"] = talib.TRIX(data.Close, timeperiod=2*n)

#Learn more about the 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA at tadoc.org.
#ULTOSC - Ultimate Oscillator

data["ULTOSC"] = talib.ULTOSC(data.High, data.Low, data.Close, timeperiod1=7, timeperiod2=14, timeperiod3=28)

#Learn more about the Ultimate Oscillator at tadoc.org.
#WILLR - Williams' %R

data["WILLR"] = talib.WILLR(data.High, data.Low, data.Close, n)


#--------------
data["C-O"] = (data["Close"] - data["Open"]) / (data["Close"])
data["C-H"] = (data["Close"] - data["High"]) / (data["Close"])

def VWAP(df):
    q = df.Volume.values
    p = df.Close.values
    VWAP=(p * q).cumsum() / q.cumsum()
    return VWAP


data["VWAP"] = VWAP(data)/data["Close"]


#Pivot Points, Supports and Resistances  
def PPSR(df):  
    PP = pd.Series((df['High'] + df['Low'] + df['Close']) / 3)  
    R1 = pd.Series(2 * PP - df['Low'])  
    S1 = pd.Series(2 * PP - df['High'])  
    R2 = pd.Series(PP + df['High'] - df['Low'])  
    S2 = pd.Series(PP - df['High'] + df['Low'])  
    R3 = pd.Series(df['High'] + 2 * (PP - df['Low']))  
    S3 = pd.Series(df['Low'] - 2 * (df['High'] - PP))  
    #psr = {'PP':PP, 'R1':R1, 'S1':S1, 'R2':R2, 'S2':S2, 'R3':R3, 'S3':S3}  
    #PSR = pd.DataFrame(psr)   
    return PP, R1, S1, R2, S2, R3, S3

data["PP"],data["R1"],data["S1"],data["R2"],data["S2"],data["R3"], data["S3"] = PPSR(data)



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



window_size = 15
data['15_min'] = 0
for i in range(0,len(data)):
    if i+window_size <= len(data):
        values = data['Close'][i:i+window_size]
        start_value = data["Close"][i] 
        max_value = max(values)
        min_value = min(values)
        if max_value > (start_value + (0.015 * start_value)):
            data['15_min'][i] = "breakout"
        elif min_value < (start_value - (0.015 * start_value)):
            data['15_min'][i] = "breakdown"
        else:
            data['15_min'][i] = "normal"
  
window_size = 30
data['30_min'] = 0
for i in range(0,len(data)):
    if i+window_size <= len(data):
        values = data['Close'][i:i+window_size]
        start_value = data["Close"][i] 
        max_value = max(values)
        min_value = min(values)
        if max_value > (start_value + (0.02 * start_value)):
            data['30_min'][i] = "breakout"
        elif min_value < (start_value - (0.02 * start_value)):
            data['30_min'][i] = "breakdown"
        else:
            data['30_min'][i] = "normal"


       
         

#
            
#Looking the  features
dataset = data[n:len(data)-(window_size -1)]
dataset = dataset.reset_index(drop = True)
dataset = dataset.fillna(0)

from collections import Counter
Counter(dataset['15_min'])

cols = ['Open', 'High', 'Low', 'Close', 'Volume',
       'MA', 'EMA', 'MOM', 'ROC', 'ATR', 'BBANDS', 'BBANDS1', 'BBANDS2', 'ADX',
       'ADXR', 'APO', 'aroondown', 'aroonup', 'AROONOSC', 'BOP', 'CCI', 'CMO',
       'DX', 'macd', 'macdsignal', 'macdhist', 'MFI', 'MINUS_DI', 'PLUS_DI',
       'PLUS_DM', 'PPO', 'ROCP', 'ROCR', 'ROCR100', 'RSI', 'slowk', 'slowd',
       'fastk', 'fastd', 'TRIX', 'ULTOSC', 'WILLR', 'C-O', 'C-H', 'VWAP', 'PP',
       'R1', 'S1', 'R2', 'S2', 'R3', 'S3']
columns = dataset.loc[:,cols].columns



#
#breakdown = dataset['5_min'] == "breakdown"
#breakout = dataset['5_min'] == "breakout"
#normals = dataset["5_min"] == "normal"
#
#import matplotlib.gridspec as gridspec
#import seaborn as sns
#grid = gridspec.GridSpec(14, 2)
#plt.figure(figsize=(15,20*4))
#
#for m, col in enumerate(dataset[columns]):
#    ax = plt.subplot(grid[m])
#    sns.distplot(dataset[col][breakout], bins = 50, color='g') #Will receive the "semi-salmon" violin
#    sns.distplot(dataset[col][normals], bins = 50, color='r') #Will receive the "ocean" color
#    sns.distplot(dataset[col][breakdown], bins = 50, color='b')
#    ax.set_ylabel('Density')
#    ax.set_title(str(col))
#    ax.set_xlabel('')
#plt.show()

#==========================


from sklearn import preprocessing


# separate the data from the target attributes
X = dataset.loc[:,cols]
y = dataset['15_min'][:]
# normalize the data attributes
X = preprocessing.normalize(X)


#=========================splitting data into train and test data===================
from sklearn.model_selection import train_test_split
seed = 7
test_size = 0.20

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, shuffle = False)
#
#====================================
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(X_train,y_train)

from imblearn.pipeline import make_pipeline as make_pipeline_imb # To do our transformation in a unique time
from imblearn.over_sampling import SMOTE
#from sklearn.pipeline import make_pipeline
#from imblearn.metrics import classification_report_imbalanced


#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score







classifier = RandomForestClassifier

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=4), \
                                   classifier(random_state=42))

smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)

#Showing the diference before and after the transformation used
print("normal data distribution: {}".format(Counter(y)))
X_smote, y_smote = SMOTE().fit_sample(X, y)
print("SMOTE data distribution: {}".format(Counter(y_smote)))

print("Confusion Matrix: ")
print(confusion_matrix(y_test, smote_prediction))

print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))

import sklearn.metrics as met
# evaluate predictions
#accuracy = accuracy_score(y_test, smote_prediction)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

print ('\nClassification Report:\n-----------------------------')
print (met.classification_report(y_test, smote_prediction))

cm = met.confusion_matrix(y_test, smote_prediction)


import seaborn as sns
ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Breakout','Breakdown', 'Normal']); ax.yaxis.set_ticklabels(['Breakout','Breakdown', 'Normal']);

compare_table = pd.DataFrame(dict(y_test = y_test, y_pred = smote_prediction)).reset_index(drop = True)
Counter(compare_table["y_test"])
Counter(compare_table["y_pred"])

evaluation_df = pd.DataFrame(X_test) 
evaluation_df = evaluation_df.reset_index(drop = True)
evaluation_df["actual"] = y_test.values
evaluation_df["predicted"] = smote_prediction

#actual breakout
eval_breakout = evaluation_df.loc[evaluation_df["actual"] == "breakout"]


#actual breakdown
eval_breakdown = evaluation_df.loc[evaluation_df["actual"] == "breakdown"]
