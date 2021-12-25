import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from mrmr import mrmr_classif
from scipy import stats
from nilearn import image as nimg
from nilearn import plotting
import matplotlib as plt

X = np.load('Reg_z.npy')
X1 = X[:,0:360]
X2 = X[:,360:17013]

Regressor_z = pd.read_csv('Regressor_z.csv')
y1 = Regressor_z['Condition1']
y1 = np.array(y1)

# Space
HEL_True = []
Predict_HEL_Anat = []
Predict_HEL_Func = []
Predict_HEL_Both = []
accuracy_HEL_Anat = np.zeros((10))
accuracy_HEL_Func = np.zeros((10))
accuracy_HEL_Both = np.zeros((10))
Anat100index_HEL = np.zeros((10,100))
Func100index_HEL = np.zeros((10,100))

# Shuffle
cv = KFold(n_splits=10, shuffle = True, random_state=5)

# GO!
i = 0
for train, test in cv.split(X):
    X_train = X[train]
    X1_train = X_train[:, 0:360]
    X2_train = X_train[:, 360:17013]

    y1_train = y1[train]

    for j in range(len(y1_train)):
        if y1_train[j] == 'HEL':
            y1_train[j] = 0
        else:
            y1_train[j] = 1
    y1_train_df = pd.Series(y1_train)
    X1_train_df = pd.DataFrame(X1_train)
    X2_train_df = pd.DataFrame(X2_train)

    # Anat Only Index
    HEL_Anat100_Index = mrmr_classif(X1_train_df, y1_train_df,K=100)
    Anat100index_HEL[i,:] = HEL_Anat100_Index

    # Func Only Index
    HEL_Func100_Index = mrmr_classif(X2_train_df, y1_train_df,K=100)
    Func100index_HEL[i, :] = HEL_Func100_Index

    HEL_Anat100 = X1_train[:,HEL_Anat100_Index]
    HEL_Func100 = X2_train[:,HEL_Func100_Index]
    HEL_200 = np.concatenate((HEL_Anat100,HEL_Func100),axis=1)

    # SVM
    X_test = X[test]
    X1_test = X_test[:, 0:360]
    X2_test = X_test[:, 360:17013]
    y1_test = y1[test]

    clf = svm.SVC(kernel='linear')
    y1_train = y1[train]

    clf.fit(HEL_Anat100, y1_train)
    prediction_Anat = clf.predict(X1_test[:,HEL_Anat100_Index])
    result = prediction_Anat == y1_test
    accuracy_HEL_Anat[i] = sum(result)/result.shape
    Predict_HEL_Anat.extend(list(prediction_Anat))

    clf.fit(HEL_Func100, y1_train)
    prediction_Func = clf.predict(X2_test[:,HEL_Func100_Index])
    result = prediction_Func == y1_test
    accuracy_HEL_Func[i] = sum(result)/result.shape
    Predict_HEL_Func.extend(list(prediction_Func))

    clf.fit(HEL_200, y1_train)
    HEL_200_test = np.concatenate((X1_test[:,HEL_Anat100_Index],X2_test[:,HEL_Func100_Index]),axis=1)
    prediction_Both = clf.predict(HEL_200_test)
    result = prediction_Both == y1_test
    accuracy_HEL_Both[i] = sum(result)/result.shape
    Predict_HEL_Both.extend(list(prediction_Both))

    HEL_True.extend(list(y1_test))

    i = i + 1

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(HEL_True,Predict_HEL_Anat,labels=["SCZ","HEL"])
confusion_matrix(HEL_True,Predict_HEL_Func,labels=["SCZ","HEL"])
confusion_matrix(HEL_True,Predict_HEL_Both,labels=["SCZ","HEL"])

# t-test
stats.ttest_1samp(a=accuracy_HEL_Anat,popmean=0.5)
stats.ttest_1samp(a=accuracy_HEL_Func,popmean=0.5)
stats.ttest_1samp(a=accuracy_HEL_Both,popmean=0.5)

stats.ttest_rel(accuracy_HEL_Anat,accuracy_HEL_Func)
stats.ttest_rel(accuracy_HEL_Anat,accuracy_HEL_Both)
stats.ttest_rel(accuracy_HEL_Func,accuracy_HEL_Both)

# accuracy report
accuracy_HEL_Anat_mean = accuracy_HEL_Anat.mean()
accuracy_HEL_Anat_std = accuracy_HEL_Anat.std()
print("%.2f" % accuracy_HEL_Anat_mean + " " + u"\u00B1" + " " + "%.3f" % accuracy_HEL_Anat_std)

accuracy_HEL_Func_mean = accuracy_HEL_Func.mean()
accuracy_HEL_Func_std = accuracy_HEL_Func.std()
print("%.2f" % accuracy_HEL_Func_mean + " " + u"\u00B1" + " " + "%.3f" % accuracy_HEL_Func_std)

accuracy_HEL_Both_mean = accuracy_HEL_Both.mean()
accuracy_HEL_Both_std = accuracy_HEL_Both.std()
print("%.2f" % accuracy_HEL_Both_mean + " " + u"\u00B1" + " " + "%.3f" % accuracy_HEL_Both_std)

np.save("Anat100index_HEL",Anat100index_HEL)
np.save("Func100index_HEL",Func100index_HEL)

# get 15 anat
(unique, counts) = np.unique(Anat100index_HEL,return_counts=True)
candidates = unique[np.logical_or(counts == 9,counts == 10)]
rank = np.zeros((len(candidates)))
for i in range(len(candidates)):
    rank[i] = np.where(Anat100index_HEL == candidates[i])[1].mean()
Anat15Index_HEL = candidates[rank.argsort()][0:15]

# get 15 func
(unique, counts) = np.unique(Func100index_HEL,return_counts=True)
candidates = unique[np.logical_or(counts == 9,counts == 10)]
rank = np.zeros((len(candidates)))
for i in range(len(candidates)):
    rank[i] = np.where(Func100index_HEL == candidates[i])[1].mean()
Func15Index_HEL = candidates[rank.argsort()][0:15]

OriginFuncIndex_HEL = np.zeros((15,2))
for k in range(15):
    for i in range(1, 183):
        for j in range(0, i):
            index = (i-1)*i/2 + j
            if index == Func15Index_HEL[k]:
                OriginFuncIndex_HEL[k,0] = i + 1
                OriginFuncIndex_HEL[k,1] = j + 1
UniqueFuncIndex_HEL = np.unique(OriginFuncIndex_HEL)

# into csv
SCZ_15CT = np.zeros((15))
HEL_15CT = np.zeros((15))
CortThickness = np.load("processdata/CortThickness.npy")

for k in range(15):
    SCZ_15CT[k] = CortThickness[y1 == "SCZ",int(Anat15Index_HEL[k])].mean()
    HEL_15CT[k] = CortThickness[y1 == "HEL", int(Anat15Index_HEL[k])].mean()

pd.DataFrame(SCZ_15CT).to_csv("processdata/SCZ_15CT.csv")
pd.DataFrame(HEL_15CT).to_csv("processdata/HEL_15CT.csv")

SCZ_15FC = np.zeros((15))
HEL_15FC = np.zeros((15))
FuncConnectivity = np.load("processdata/FuncConnectivity.npy")
FuncConnectivity = np.mean(FuncConnectivity,axis=1)

for k in range(15):
    SCZ_15FC[k] = FuncConnectivity[y1 == "SCZ",int(OriginFuncIndex_HEL[k,0] -1),int(OriginFuncIndex_HEL[k,1] -1)].mean()
    HEL_15FC[k] = FuncConnectivity[y1 == "HEL",int(OriginFuncIndex_HEL[k,0] -1),int(OriginFuncIndex_HEL[k,1] -1)].mean()

pd.DataFrame(SCZ_15FC).to_csv("processdata/SCZ_15FC.csv")
pd.DataFrame(HEL_15FC).to_csv("processdata/HEL_15FC.csv")

# plot 15 anat
parcel_file = 'rois/HCP-MMP_1mm.nii.gz'
Glasser_180 = nimg.load_img(parcel_file)

for k in range(len(Anat15Index_HEL)):
    if Anat15Index_HEL[k] > 180:
        Anat15Index_HEL[k] = Anat15Index_HEL[k] + 20

Final_Anat_HEL = nimg.math_img('0*a', a=Glasser_180)

for i in range(len(Anat15Index_HEL)):
    Formula1 = 'a == ' + str(Anat15Index_HEL[i])
    roi_mask = nimg.math_img(Formula1, a=Glasser_180)
    Formula2 = 'a + (' + str(round(SCZ_15CT[i],3)) + ' * b)'
    Final_Anat_HEL = nimg.math_img(Formula2, a=Final_Anat_HEL, b=roi_mask)

plotting.plot_glass_brain(Final_Anat_HEL)
plt.savefig("results/HEL.png")