#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:42:11 2021

@author: bweborg
"""

from lyon.calc import LyonCalc
import numpy as np
from scipy.io import wavfile as wav
from os import listdir
from os.path import isfile, join, exists
import pandas as pd

def getFrequencies(audio_dir, outputFile, decimation_factor = 64):
    file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
    
    #stuff needed for loop
    calc = LyonCalc()
    specimen = 0
    
    #set up data array
    sample_rate, waveform = wav.read(audio_dir + file_names[0])
    waveform = waveform.astype(float)
    cols = np.array(calc.lyon_passive_ear(waveform, sample_rate, decimation_factor, step_factor = 0.19)).shape[1]
    data = np.ones((0,cols+2))
    
    print("Processing files...")
    for file_name in file_names:
        target = file_name[0]
        audio_path = audio_dir + file_name
        sample_rate, waveform = wav.read(audio_path)
        waveform = waveform.astype(float)
        
        coch = np.array(calc.lyon_passive_ear(waveform, sample_rate, decimation_factor, step_factor = 0.19))
        
        group = np.full((coch.shape[0], 1), specimen)
        targets = np.full((coch.shape[0], 1), target)
        specimen = specimen + 1
        coch = np.append(group, coch, axis=1)
        coch = np.append(coch, targets, axis=1)
        data = np.append(data, coch, axis=0)
        
        #if specimen == 50:
        #    break
    print("Printing to CSV")
    pd.DataFrame(data).to_csv(outputFile, header=False, index=False)
    print("Done")
    return

#train a reservoir to output sine wave twice the initial amplitude
def main():
    audio_dir = 'recordings/'
    outputFile = 'data.csv'
    if not(exists(outputFile)):
        data = getFrequencies(audio_dir,outputFile, 64)
    data = pd.read_csv(outputFile)
        
    print(data.head())
    

    return

if __name__ == "__main__":
    main()