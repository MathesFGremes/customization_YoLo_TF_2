from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import random

objects = OrderedDict()
'''
id = 1
for i in np.arange(10):
    j = i*10
    x = random.randint(0,100)
    y = random.randint(0,100)
    objects[id] = (x, y)
    id = id + 1
'''
objects[0] = (79,44)
objects[4] = (95,5)
objects[9] = (13,99)
objects[12] = (25,59)
objects[15] = (84,59)
objects[40] = (85,45)
objects[42] = (94,2)
objects[48] = (73,68)
objects[52] = (17,35)
objects[53] = (82,62)
'''
objectIDs = list(objects.keys())
objectCentroids = list(objects.values())

#print(objectIDs)
#print(objectCentroids)
#print(np.array(objectCentroids))

D = dist.cdist(np.array(objectCentroids), np.array(objectCentroids))
DZ = (D + (D == 0)*500)
DZ = D
rows = D.min(axis=1).argsort()
rowsZ = DZ.min(axis=1).argsort()
colsZ = DZ.argmin(axis=1)[rowsZ]

np.set_printoptions(precision=2)
#print(D)
#print()
#print(rowsZ)
#print()
#print(colsZ)
#print()
#print(DZ[rowsZ])
print(DZ)
print()
print(DZ.min(axis=1))
print()
print(DZ.argsort())
print()
print(DZ.argsort()[:,0])
print()
A = DZ.argsort()[:,0].astype(int)


print(np.array(objectIDs)[A])
print()
print(DZ.argsort())
argsort = DZ.argsort()
'''

objectIDs = list(objects.keys())
objectCentroids = list(objects.values())
DZ = dist.cdist(np.array(objectCentroids), np.array(objectCentroids))
argsort = DZ.argsort()

dMaxNeighbor = 10

for i in np.arange(len(objectIDs)):
    #print("id objeto: ", objectIDs[i])
    for j in np.arange(len(objectIDs)-1)+1:
        if DZ[i, argsort[i,j]] < dMaxNeighbor:
            print("id objeto: ", objectIDs[i])
            print("id vizinhos: ", objectIDs[argsort[i,j]])
            print("D vizinho: ", DZ[i, argsort[i,j]])
            print()
#print(D + (D == 0)*1000)