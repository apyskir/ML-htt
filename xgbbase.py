#!/usr/bin/env python

import time
import xgboost as xgb
import pandas as pd
from sklearn import *
import numpy as np

fileName = "RootAnalysis_AnalysisMuTau_pandas_big"
print "start read"
start = time.time()
data = pd.read_pickle(fileName)
end = time.time()
print "end read: ", end - start, " s"
#print(data.head())
signalNumbers = [5,6,7,8,9]
bkgNumbers=[1,2,3,4]
print "start convert"
data.loc[data["sampleNumber"].isin(bkgNumbers), "signal"] = 0
data.loc[data["sampleNumber"].isin(signalNumbers), "signal"] = 1
data.loc[data["sampleNumber"]==0, "signal"] = -1
print "end convert"

bkg = data[data["sampleNumber"].isin(bkgNumbers)]
signal = data[data["sampleNumber"].isin(signalNumbers)]
#signal = data[data["npv"]>20]	#test
#bkg = data[data["npv"]<20]	#test

train = data[data["sampleNumber"] > 0]
test = data[data["sampleNumber"] == 0]

#print "train size: ", train.shape[0]
#print "test size: ", test.shape[0]

def splitIntoTTV(df, trainingFraction, testFraction):
    trainSample = df.sample(frac = trainingFraction)
    size = float(df.shape[0])
    #print "total size: ", df.shape[0]
    df = pd.concat([df, trainSample]).drop_duplicates(keep=False)
    testSample = df.sample(frac = testFraction/(1 - trainingFraction))
    valSample = pd.concat([df, testSample]).drop_duplicates(keep=False)
    #print "train size: ", trainSample.shape[0], trainSample.shape[0]/size
    #print "test size: ", testSample.shape[0], testSample.shape[0]/size
    #print "val size: ", valSample.shape[0], valSample.shape[0]/size
    return [trainSample, testSample, valSample]

#a = splitIntoTTV(signal[signal["sampleNumber"]==5], 0.6, 0.2)
#print a[0].shape[0], a[1].shape[0], a[2].shape[0]

def splitIntoTT(df, trainingFraction):
    trainSample = df.sample(frac = trainingFraction)
    size = float(df.shape[0])
    #print "total size: ", df.shape[0]
    testSample = pd.concat([df, trainSample]).drop_duplicates(keep=False)
    #print "train size: ", trainSample.shape[0], trainSample.shape[0]/size
    #print "test size: ", testSample.shape[0], testSample.shape[0]/size
    return [trainSample, testSample]

#a = splitIntoTT(signal[signal["sampleNumber"]==5], 0.6)
#print a[0].shape[0], a[1].shape[0]


def splitTrainIntoTTV(df, trainingFraction, testFraction):
    #assumption:
    trainSample = pd.DataFrame()
    testSample = pd.DataFrame()
    valSample = pd.DataFrame()
    for i in df["sampleNumber"].value_counts().index:
        x = splitIntoTTV(df[df["sampleNumber"]==i], trainingFraction, testFraction)
        #print i
        trainSample = pd.concat([trainSample, x[0]])
        testSample = pd.concat([testSample, x[1]])
        valSample = pd.concat([valSample, x[2]])
    return [trainSample, testSample, valSample]

#a = splitTrainIntoTTV(train[train["sampleNumber"].isin([8,9])], 0.6, 0.2)
#print a[0].shape[0], a[1].shape[0], a[2].shape[0]


def splitTrainIntoTT(df, trainingFraction):
    #assumption:
    trainSample = pd.DataFrame()
    testSample = pd.DataFrame()
    for i in df["sampleNumber"].value_counts().index:
        x = splitIntoTT(df[df["sampleNumber"]==i], trainingFraction)
        #print i
        trainSample = pd.concat([trainSample, x[0]])
        testSample = pd.concat([testSample, x[1]])
    return [trainSample, testSample]

#a
#trainSample, testSample = splitTrainIntoTT(train[train["sampleNumber"].isin([8,9])], 0.6)
#print trainSample.shape[0], testSample.shape[0]
#print a[0].shape[0], a[1].shape[0]

def getNewArytmList(mid, diff, shrinkage, name):
    newList = [value for value in list(np.arange(mid - (shrinkage-1)*diff/shrinkage, mid+(shrinkage-1+0.1)*diff/shrinkage, diff/shrinkage)) if value >= 0]
    print "diff:", diff, "diff/shrinkage:", diff/shrinkage
    print "Nowa lista dla", name, newList
    if name in ["subsample", "colsample_bytree"]:
        newList = [x for x in newList if x<=1]
        print "Nowa lista po poprawkach dla", name, newList
    if name in ["max_depth",]:
        for x in newList:
            print abs(x-round(x))
            print abs(x-round(x))<0.0000001
        newList = [int(round(x)) for x in newList if abs(x-round(x))<0.0000001]
        print "Nowa lista po poprawkach dla", name, newList
    return newList
