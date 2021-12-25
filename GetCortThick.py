import pandas as pd
import numpy as np

# data = pd.read_excel('thickness/sub_01_thickness.xlsx')
# roiname = data['StructName']
# roiname.to_csv('rois/RoiName.csv')

roiname = list(pd.read_csv('rois/RoiName.csv')['StructName'])

participants = pd.read_excel('fmriprep/participants.xlsx')
subindex = np.array(participants['subindex'])
condition = participants['Condition3']

i = 0
CortThick = pd.DataFrame(np.zeros((98,360)),columns=roiname)

for sub in subindex:
    if sub < 10:
        path = "thickness/sub_0" + str(sub) + "_thickness.xlsx"
        data = pd.read_excel(path)
        value = pd.Series(np.array(data['Mean']), index=list(roiname))
        CortThick.loc[i] = value
        i = i + 1
    else:
        path = "thickness/sub_" + str(sub) + "_thickness.xlsx"
        data = pd.read_excel(path)
        value = pd.Series(np.array(data['Mean']), index=list(roiname))
        CortThick.loc[i] = value
        i = i + 1

CortThick.insert(0,"SubIndex",subindex,True)
CortThick.insert(1,"Condition",condition,True)

CortThick.to_csv('processdata/CortThick.csv')

# convert into nparray
AnatomicalRegressor = pd.read_csv('processdata/CortThick.csv')
CortThickness = np.array(AnatomicalRegressor[AnatomicalRegressor.columns[3:363]])
np.save("processdata/CortThickness",CortThickness)

