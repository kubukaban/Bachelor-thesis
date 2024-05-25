import os
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
data = {}
x = []
y = []
for i in tqdm(range(0,221,20)):
    vertices = []
    faces = []
    text = '000'+str(i)
    with open('frame'+text[-4:]+'.obj', 'r') as file:
        file_contents = file.read()
        count = file_contents.count('f')
        file.seek(0)
        for line in file:
            if line.startswith('v '):
              vertex = list(map(float, line.split()[1:]))
              vertices.append(vertex)
            elif line.startswith('f '):
               points = line.split()[1:]
               try:
                  triangle = [int(re.search(r'\d+(?=(//))',x).group()) for x in points]
               except AttributeError:
                   try:
                       triangle = [int(re.search(r'\d+(?=(/))',x).group()) for x in points]
                   except AttributeError:
                       triangle = list(map(int,points))
               faces.append(triangle)
        faces = pd.DataFrame(faces,columns = ["A","B","C"])
        vertices = pd.DataFrame(vertices, columns = ['x','y','z'])
        
        result = np.zeros((6, len(faces)))
        sin_alpha = np.zeros(len(faces))
        sin_beta = np.zeros(len(faces))
        sin_gamma = np.zeros(len(faces))
        dlhsie = np.zeros(len(faces))
        kratsie = np.zeros(len(faces))
        thety0 = np.zeros(len(faces))
        thetymax = np.zeros(len(faces))
        alphas = []
        betas = []
        gammas = []
        areas = np.zeros(len(faces))
        
        for j in range(len(faces)):
            point_A = vertices.iloc[faces.iloc[j, 0] - 1]
            point_B = vertices.iloc[faces.iloc[j, 1] - 1]
            point_C = vertices.iloc[faces.iloc[j, 2] - 1]

            c = np.sqrt(np.sum((point_A - point_B) ** 2))
            a = np.sqrt(np.sum((point_B - point_C) ** 2))
            b = np.sqrt(np.sum((point_C - point_A) ** 2))

            alpha = np.arccos((-a ** 2 + b ** 2 + c ** 2) / (2 * c * b))
            beta = np.arccos((a ** 2 - b ** 2 + c ** 2) / (2 * a * c))
            gamma = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
            alphas.append(alpha)
            betas.append(beta)
            gammas.append(gamma)
            dlhsie[j] = max(a, b, c)
            kratsie[j] = min(a, b, c)

            thety0[j] = min(alpha, beta, gamma)
            thetymax[j] = max(alpha, beta, gamma)
            sin_alpha[j] = np.sin(alpha)
            sin_beta[j] = np.sin(beta)
            sin_gamma[j] = np.sin(gamma)
            
            s = (a+b+c)/2
            areas[j] = np.sqrt(s*(s-a)*(s-b)*(s-c))
        areas = areas/np.min(areas) #normalizacia ploch
        result[0, :] = (sin_alpha + sin_beta + sin_gamma) / (2 * sin_alpha * sin_beta * sin_gamma) # ro
        result[1, :] = dlhsie / kratsie # tau
        result[2, :] = 1 / (sin_alpha + sin_beta + sin_gamma) # nu
        result[3, :] = (np.sin(thety0) + np.sin(thetymax) + np.sin(thety0 + thetymax)) / (
                    np.sin(thety0) * np.sin(thety0 + thetymax)) # iota
        result[4, :] = 1 / (2 * np.sin(thetymax)) # omega
        result[5, :] = areas
        
        faces['pomer_polomerov'] = result[0, :] / 2
        faces['pomer_extrem_stran'] = result[1, :]
        faces['pomer_polomer_polobvod'] = result[2, :] * 3 * np.sqrt(3) / 2
        faces['pomer_stran'] = result[3, :] / (2 * np.sqrt(3))
        faces['pomer_opisana_strana'] = result[4, :] * 2
        faces['obsahy'] = result[5]
        data[i] = [count,np.median(faces['pomer_polomerov']),np.median(faces['pomer_extrem_stran']),
                   np.median(faces['pomer_polomer_polobvod']),np.median(faces['pomer_stran']),
                   np.median(faces['pomer_opisana_strana']),np.median(faces['obsahy']),np.var(faces['pomer_polomerov']),np.var(faces['pomer_extrem_stran']),
                   np.var(faces['pomer_polomer_polobvod']),np.var(faces['pomer_stran']),
                   np.var(faces['pomer_opisana_strana']),np.var(faces['obsahy'])
                   ]    
        
        
    x.append(i)
    y.append(c)
df = pd.DataFrame.from_dict(data, orient='index')
df.index.name = 'Index'
df = df.rename(columns={0: 'Pocet', 1: 'median pomer_polomerov', 2: 'median pomer_extrem_stran',
                        3: 'median pomer_polomer_polobvod', 4 : 'median pomer_stran', 5 : 'median pomer_opisana_strana',
                        6 : 'median obsahy',  7: 'rozptyl pomer_polomerov', 8: 'rozptyl pomer_extrem_stran',
                        9: 'rozptyl pomer_polomer_polobvod', 10 : 'rozptyl pomer_stran', 11 : 'rozpyl pomer_opisana_strana',
                        12 : 'rozpyl obsahy'})
print(df)
df.to_csv('data2.csv')