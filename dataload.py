# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:40:49 2021

@author: 1999a
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import sklearn

labels = []
wordNames = ["a", "and", "as", "at", "be", "but", "by", "do", "for", "from", 
             "have", "he", "her", "his", "i", "in", "it", "not", "of", "on", 
             "or", "say", "she", "that", "the", "they", "this", "to", "we",
             "with", "you"]
numWords = len(wordNames) * 25
words = np.zeros((numWords,88000))
mfcc = np.zeros((numWords,20,172))
for k in range(len(wordNames)):
    print(k)
    for i in range(25):
        labels.append(wordNames[k])
        if i < 9:
            words[i], _= librosa.load("words/and" + "0" + str(i+1) + ".wav", sr=44000)
        else:
            words[i], _= librosa.load("words/and" + str(i+1) + ".wav", sr=44000)
        
for i in range(numWords):
    mfcc[i] = librosa.feature.mfcc(y=words[i], sr=44000)


print(mfcc)
