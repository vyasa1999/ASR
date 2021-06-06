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
y, sr = librosa.load("words/do20.wav", sr=44000)
labels = []
words = np.zeros((25,88000))
for i in range(24):
    labels.append("do")
    if i < 9:
        words[i], _= librosa.load("words/do" + "0" + str(i+1) + ".wav", sr=44000)
    else:
        words[i], _= librosa.load("words/do" + str(i+1) + ".wav", sr=44000)
        

mfcc = librosa.feature.mfcc(y=words[0], sr=44000)
#for i in range(24):
 #   mfcc[i] = librosa.feature.mfcc(y=words[i], sr=44000)

librosa.display.specshow(mfcc, sr=sr, x_axis='time')
