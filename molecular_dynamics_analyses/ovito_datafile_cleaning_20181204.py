# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 20:01:16 2018

@author: maozhang
"""

# this script is designed to read and clean the data generated by ovito


# import the requred moduli
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# check the number of files with suffix '.data' in the current directory
path = os.getcwd()
number = 0
for root, dirname, filenames in os.walk(path):
    for filename in filenames:
        if os.path.splitext(filename)[1] == '.data':
            with open(filename) as f:
                lines = f.readlines()
            timestep = lines[1]
            num_atoms = lines[3]
            x_range, y_range, z_range = lines[5:8]
            items = lines[8].split(' ')
            columns = items[2:]
            tables = lines[9:]

            tmpname = filename[:-5] + '.tmp'
            txtname = filename[:-5] + '.txt'
            with open(tmpname, 'w') as nf:
                for char in columns:
                    nf.write(char+' ')
                for lines in tables:
                    nf.write(lines)

            origin_data = pd.DataFrame(pd.read_csv(tmpname,sep='\s+'))

            origin_data['VoronoiIndices'] = '<'+origin_data['VoronoiIndex3'].map(str) +' '+origin_data['VoronoiIndex4'].map(str) +' '+origin_data['VoronoiIndex5'].map(str) +' '+origin_data['VoronoiIndex6'].map(str)+'>'
            origin_data.rename(columns={'id':'AtomID','type':'AtomType','Coordination':'CoordinationNumber', 'x':'PositionX','y':'PositionY','z':'PositionZ','c_myke':'KineticEnergy','c_ppe':'PotentialEnergy','c_ps1':'StressXX','c_ps2':'StressYY','c_ps3':'StressZZ','c_ps4':'StressXY','c_ps5':'StressXZ','c_ps6':'StressYZ'},inplace=True)
            data = origin_data.drop(['VoronoiIndex3', 'VoronoiIndex4', 'VoronoiIndex5', 'VoronoiIndex6'], axis='columns')
            data.to_csv(txtname, sep= '\t')

# delete the tmp file generated during data cleaning
for root, dirs, files in os.walk(path):
    for name in files:
        if(name.endswith('.tmp')):
            os.remove(os.path.join(root,name))



