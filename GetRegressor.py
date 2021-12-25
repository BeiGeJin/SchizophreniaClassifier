import numpy as np
import scipy.stats as stats

# flatten FC
FuncConnectivity = np.load('processdata/FuncConnectivity.npy')
FuncConnectivity_taskmean = np.mean(FuncConnectivity,axis=1)
FuncConnectivity_flatten = np.zeros((98,16653))

l = FuncConnectivity_taskmean.shape[1]
for k in range(0,98):
    for i in range(1,l):
        for j in range(0,i):
            index = (i-1)*i/2 + j
            index = int(index)
            FuncConnectivity_flatten[k,index] = FuncConnectivity_taskmean[k,i,j]

# Combine Regressor
CortThickness = np.load('processdata/CortThickness.npy')

Reg = np.concatenate((CortThickness, FuncConnectivity_flatten),axis=1)
np.save("processdata/Reg",Reg)

Reg_z = stats.zscore(Reg,axis=0)
np.save("processdata/Reg_z",Reg_z)
