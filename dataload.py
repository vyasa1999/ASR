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
from sklearn.preprocessing import OneHotEncoder
import sys
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
   
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}%".format( "#"*block + "-"*(barLength-block), progress*100,)
    sys.stdout.write(text)
    sys.stdout.flush()
def loadPreprocess():
    labels = []
    wordNames = ["a", "and", "as", "at", "be", "but", "by", "do", "for", "from", 
                 "have", "he", "her", "his", "i", "in", "it", "not", "of", "on", 
                 "or", "say", "she", "that", "the", "they", "this", "to", "we",
                 "with", "you"]
    numWords = len(wordNames) * 25
    words = np.zeros((numWords,88000))
    mfcc = np.zeros((numWords,20,172))
    print("Preprocessing Started, percent complete:")
    for k in range(len(wordNames)):
        update_progress((k+1)/len(wordNames))
        for i in range(25):
            labels.append(wordNames[k])
            if i < 9:
                words[i+(k*25)], _= librosa.load("words/" + wordNames[k] + "0" + str(i+1) + ".wav", sr=44000)
            else:
                words[i+(k*25)], _= librosa.load("words/" + wordNames[k] + str(i+1) + ".wav", sr=44000)
            
    for i in range(numWords):
        mfcc[i] = librosa.feature.mfcc(y=words[i], sr=44000)
    
    
    return mfcc, labels
