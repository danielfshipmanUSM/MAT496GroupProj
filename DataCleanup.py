# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:05:17 2021

@author: Daniel Shipman

This program should cleanup the True.csv and Fake.csv and produce one NewsData.csv

The format of NewsData should be as followed

Unique String 1 | Unique String 2 | ... | Unique String p | True
"""

import numpy as np
import pandas as pd 
from bisect import bisect_left as bl
from matplotlib import pyplot as plt
import re

# StringWords = ["There are few daisies", "The sky is blue tonight", "I can't tell where I am"]
# StringWords = ["AAADWDWDW WDAWDA WEDAD A A", "DWAD WD A OE PC E A A A", "OKD E O KL KS P KE "]

"""
Couple manipulations to do to StringWords, namely, need to remove all '.' and ',', Additionally,

Many (@names) are in the data, since all the fake news is from twitter in this dataset, these should be removed.
another problem is typos. This problem wont be adressed since a typo would be a good giveaway that the
sources is fake. Additionally all words should be lowercase. Additionally we're removing all numbers from 
the data, since the fake data contains many number and text combinations. However, this needs to be explored deeper.

I decided to remove all the numbers from the files, additionally there are still too many predictors

p ~ 4*n so some more post processing of the data is required. However that'll come at the next stage.

If we narrow the focus of the question to "can frequency of words classify" rather than "what words classify"
we could only use unique words that are both in Fake and True. This would allow us greatly reduce 
the number of predictors in our final dataset. 

"""

Raw_Fake_With_Dupes = pd.read_csv("Data/Fake.csv")
Raw_True_With_Dupes = pd.read_csv("Data/True.csv")

# REMOVE DUPLICATES FROM FAKE AND TRUE
Raw_Fake = Raw_Fake_With_Dupes.drop_duplicates(subset=['text'])
Raw_True = Raw_True_With_Dupes.drop_duplicates(subset=['text'])

Raw_Fake = Raw_Fake.reset_index(drop=True)
Raw_True = Raw_True.reset_index(drop=True)

wordlist = []
lower_wordlist = []




n = 10000 # Raw_Fake.shape[0]
uniqueWords = []
uniqueWordsFalse = []
uniqueWordsTrue = []
combinedWords = []
numcount = []

#binary search to see if element is in list. 
def BinSearch(a , x):
    i = bl(a, x)
    if i != len(a) and a[i] == x:
        return i
    elif a[len(a) - 1] == x:
        return i
    else:
        return False
    

#Standardize the data to scale from 0-1 
# Not a great algorithm but it should work. O(n^2)...
def StandardizeScale(df,verbose=False):
    stand_df = pd.DataFrame.copy(df)
    max_val = 0
    k=1
    for i in stand_df:
        max_col = np.max(df[i])
        if max_col >= max_val:
            if verbose:
                print("Finding Max Value {0} / {1}".format(k,len(df.columns)*len(df)))
                k += 1
            max_val = max_col
    k = 1
    for i in stand_df:
        stand_df[i] = stand_df[i].astype(np.float32)
        for j in range(len(df[i])):
            if verbose:
                print("Scaling all values {0} / {1}".format(k,len(df.columns)*len(df)))
                k += 1
            stand_df.loc[:,(i,j)] = float(stand_df.loc[:,(i,j)] ) / max_val
    return stand_df


# Find outliers
# Not the best algorithm since it searches through

def FindOutliers(df, cutoff, verbose=False,standardize=True, right=True):
    
    #Quantiles arent actually quantiles. Need to determine the range a percentage of the data is in. 
    # 
    if standardize:      
        stand_df = StandardizeScale(df,verbose=verbose)
    else:
        stand_df = pd.DataFrame.copy(df)
    outlier = []
    for i in stand_df:
        for j in range(len(stand_df[i])):
            if stand_df[i][j] >= cutoff and right:
                outlier.append(i)
                break
            elif stand_df[i][j] <= cutoff and not right:
                outlier.append(i)
                break
    return outlier
    #CREATE COLUMN NAMES BASED ON UNIQUE WORDS 
    

  
for i in range(n):
    StringWords = re.split(" ", re.sub("[^a-zA-Z]", " ", Raw_Fake['text'][i]))
    for j in StringWords:
        StringWords.remove(j)
        lower_wordlist.append(j.lower())
    temp_words = np.unique(lower_wordlist)
    lower_wordlist = []
    uniqueWordsFalse = np.unique(np.append(uniqueWordsFalse,temp_words))
    uniqueWords = np.unique(np.append(uniqueWords,temp_words))
    print("Creating unique list from False {0} / {1}".format(i+1,n))
    
for i in range(n):
    StringWords = re.split(" ", re.sub("[^a-zA-Z]", " ", Raw_True['text'][i]))
    for j in StringWords:
        StringWords.remove(j)
        lower_wordlist.append(j.lower())
    temp_words = np.unique(lower_wordlist)
    lower_wordlist = []
    uniqueWordsTrue = np.unique(np.append(uniqueWordsTrue,temp_words))
    uniqueWords = np.unique(np.append(uniqueWords,temp_words))
    print("Creating unique list from True {0} / {1}".format(i+1,n))

uniqueWords = np.append(uniqueWords, "RealNews")
    ##### CREATE DATAFRAME #########
    
DataSet = pd.DataFrame(0, index=np.arange(2*n), columns=uniqueWords)


    ##### LOAD IN COUNTS #####
for i in range(n):
    StringWords = re.split(" ", re.sub("[^a-zA-Z]", " ", Raw_Fake['text'][i]))
    for j in StringWords:
        StringWords.remove(j)
        lower_wordlist.append(j.lower())
    temp_words = np.unique(lower_wordlist, return_counts=True)
    lower_wordlist = []
    for j in range(len(temp_words[0])):
        DataSet.iloc[i][temp_words[0][j]] = temp_words[1][j]+DataSet.iloc[i][temp_words[0][j]]
    print("Counting sum of unique from False {0} / {1}".format(i,n))
    
for i in range(n):
    StringWords = re.split(" ", re.sub("[^a-zA-Z]", " ", Raw_True['text'][i]))
    for j in StringWords:
        StringWords.remove(j)
        lower_wordlist.append(j.lower())
    temp_words = np.unique(lower_wordlist, return_counts=True)
    lower_wordlist = []
    DataSet.iloc[i+n]["RealNews"] = 1
    for j in range(len(temp_words[0])):
        DataSet.iloc[i+n][temp_words[0][j]] = temp_words[1][j]+DataSet.iloc[i+n][temp_words[0][j]]
    print("Counting sum of unique from True {0} / {1}".format(i,n))
    

# Debug code
# uniqueWordsFalse = ["bee" , "knees", "trees", "boats", "bears", "bulls"]
# uniqueWordsTrue = ["bee" , "boat", "trees"]

np.sort(uniqueWordsFalse)
np.sort(uniqueWordsTrue)
combinedWords = []

k = 0
for i in uniqueWordsFalse:
    k = k+1
    x = BinSearch(uniqueWordsTrue, i)
    print("Checking correlating words {0} / {1}".format(k,len(uniqueWordsFalse)))
    if x != False:
        combinedWords.append(i)

combinedWords = np.append(combinedWords, "RealNews")
combinedData = pd.DataFrame(0, index=np.arange(2*n), columns=combinedWords)

k = 1
for i in combinedData:
    combinedData[i] = DataSet[i]
    print("Filling combined Data {0} / {1}".format(k,len(combinedData.columns)))
    k += 1 
combinedData['RealNews'] = DataSet['RealNews']
#StandardizeScale(combinedData)
DataSet.to_csv("Formatted_Data_{0}.csv".format(n))
combinedData.to_csv("Formatted_Data_Union_{0}.csv".format(n))

TotalCount = []

for i in DataSet:
    count_tuple = [np.sum(DataSet[i]), i]
    TotalCount.append(count_tuple)
 
sortedTotalCount = TotalCount
sortedTotalCount.sort(reverse=True)
plt.hist(sortedTotalCount[1][:10],log=True)

X = combinedData.loc[:,combinedData.columns != 'RealNews']
X = StandardizeScale(X, verbose=True)
# outliers = FindOutliers(X, 0.1, standardize=False)

# X_Cut = X.drop(columns=outliers)
# plt.hist(X_Cut)
print(DataSet)