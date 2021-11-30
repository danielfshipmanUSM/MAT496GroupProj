#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 10:38:13 2021

@author: Daniel Shipman

This program fits the Naive Bayes Classifier to the Scrubbed Dataset, since the data values are all discrete wordvectors,
the Multinomial Naive Bayes Classifier should work well. The assumptions for the MNBC is:
    - Mutually Independent Data Entries

This classifier tends to perform well with discrete values and doesnt suffer too much from the curse of dimensionality.
Considering we have over 22000 predictors, this would be a highly valueable feature. It should be noted that some classifiers
such as random trees or boosted trees tend to outperform this classifier. 
"""


import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt


Data = pd.read_csv("//home//daniels//Documents//School//MAT496GroupProj//Formatted_Data_Union_3000.csv")
wordlisted_data = pd.read_csv("//home//daniels//Documents//School//MAT496GroupProj//Formatted_Data_Union_Wordlist3000.csv")

X = wordlisted_data.drop(columns='RealNews')
y = wordlisted_data['RealNews']


alphaOpt = 0.05 # determined on 95 values between 0.05 and 1
depthOpt = 8
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
# clf = MultinomialNB(alpha=alphaOpt)
clf = RandomForestClassifier(max_depth=depthOpt, random_state=0)
y_pred = clf.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=clf.classes_)
disp.plot()


smoothing_steps = np.arange(0.05,1,0.01)

# Curious to see what the top 10 most influential words are. 


smooth_acc = []
for i in smoothing_steps:
    clf = MultinomialNB(alpha=i)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    smooth_acc.append((y_test != y_pred).sum()/X_test.shape[0])
    
depth_len = np.arange(1,10)
depth_acc = []
for i in depth_len:
    clf = clf = RandomForestClassifier(max_depth=i, random_state=0)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    depth_acc.append((y_test != y_pred).sum()/X_test.shape[0])

np.argmin(depth_acc)
plt.scatter(depth_len,depth_acc)

n = X.shape[0]
k = 10
kf = KFold(n_splits=k)
kf.get_n_splits(X)

deltaAcclen = np.arange(1, 11)
deltaAcc = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    deltaAcc.append((y_test != y_pred).sum()/X_test.shape[0])

plt.scatter(deltaAcclen,deltaAcc)
