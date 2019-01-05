# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:44:31 2018

@author: maozhang
"""

# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


path = os.getcwd()
number = 0
for root, dirname, filenames in os.walk(path):
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.txt':
            number += 1
            temp = int(filename[9:12])

steps = number * 100

for step in range(0,steps,100):
    estrain = step * 0.01 * 0.042
    txt_name = 'ZrCuO_NG_%iK_%i.txt'%(temp,step)

    df00 = pd.DataFrame(pd.read_csv(txt_name, sep='\t'))
    ShearStrainStd = df00.ShearStrain.std()
    ShearStrainMean = df00.ShearStrain.mean()
    print(estrain, ShearStrainStd, ShearStrainMean)

    with open('ZrCuO_NG_%iK_ShearStrainLocalization.txt'%temp,'a+') as f:
        var = 'estrain' +'\t' + 'ShearStrainStd' +'\t' + 'ShearStrainMean' +'\t' +'\n'
        value = '%.3f\t%.5f\t%.5f\n'%(estrain,ShearStrainStd,ShearStrainMean)
        if step == 0:
            f.write(var)
            f.write(value)
        else:
            f.write(value)