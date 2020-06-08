import numpy as np
import os
import re 
file = os.listdir('data/mobility/')    
files = []
vocabulary = []
for item in file:
    files.append(item.split('_')[0]+'.txt')


files.sort(key= lambda x:int(x[:-4]))
files

for i in range(len(files)):
     files[i]= files[i].split('.')[0] + '_arr.txt'


for i in range(len(files)):
    info = open('data/mobility/'+files[i],'r') #读取文件内容
    data = info.read().splitlines()
    data = ''.join(data)
    vocabulary.append(data)
    info.close()

for i in range(len(vocabulary)):
    vocabulary[i] = vocabulary[i].split()

for i in range(len(vocabulary)):
    for j in range(len(vocabulary[i])):
        vocabulary[i][j] = vocabulary[i][j].split('_')
        for k in range(len(vocabulary[i][j])):
            vocabulary[i][j][k] = int(re.search('\d+',vocabulary[i][j][k]).group())

m = np.zeros((951,1008,24))
for i in range(len(vocabulary)) :
    for j in vocabulary[i] :
        z = i
        x = j[0]
        y = j[2] 
        m[z][x][y] = m[z][x][y] + 1

from itertools import chain
data = [[] for i in range(len(m))]
for i in range(len(m)):
    data[i].extend(list(chain(*m[i])))


dataqc = []
for item in data :
    for site in item :
        dataqc.append(site)
dataws = dataqc
print(len(dataws))

dataqc = list(set(dataqc))
datadic =  {wIdx:dataqc[wIdx] for wIdx in range(len(dataqc))}

ws = []

for item in dataws :
    for site in datadic :
        if(item == datadic[site]):
            ws.append(site)
print(len(ws))
sz = open('ws.txt','w')
sz.write(str(ws))
sz.close()






