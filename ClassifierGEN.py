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
y2 = Regressor_z['Condition2']
y2 = np.array(y2)

# Space
GEN_True = []
Predict_GEN_Anat = []
Predict_GEN_Func = []
Predict_GEN_Both = []
accuracy_GEN_Anat = np.zeros((10))
accuracy_GEN_Func = np.zeros((10))
accuracy_GEN_Both = np.zeros((10))
Anat100index_GEN = np.zeros((10,100))
Func100index_GEN = np.zeros((10,100))

# Shuffle
cv = KFold(n_splits=10, shuffle = True, random_state=3)

# GO!
i = 0
for train, test in cv.split(X):
    X_train = X[train]
    X1_train = X_train[:, 0:360]
    X2_train = X_train[:, 360:17013]

    y2_train = y2[train]

    for j in range(len(y2_train)):
        if y2_train[j] == 'CON(&SIB)':
            y2_train[j] = 0
        else:
            y2_train[j] = 1
    y2_train_df = pd.Series(y2_train)
    X1_train_df = pd.DataFrame(X1_train)
    X2_train_df = pd.DataFrame(X2_train)

    # Anat Only Index
    GEN_Anat100_Index = mrmr_classif(X1_train_df, y2_train_df,K=100)
    Anat100index_GEN[i,:] = GEN_Anat100_Index

    # Func Only Index
    GEN_Func100_Index = mrmr_classif(X2_train_df, y2_train_df,K=100)
    Func100index_GEN[i, :] = GEN_Func100_Index

    GEN_Anat100 = X1_train[:,GEN_Anat100_Index]
    GEN_Func100 = X2_train[:,GEN_Func100_Index]
    GEN_200 = np.concatenate((GEN_Anat100,GEN_Func100),axis=1)

    # SVM
    X_test = X[test]
    X1_test = X_test[:, 0:360]
    X2_test = X_test[:, 360:17013]
    y2_test = y2[test]

    clf = svm.SVC(kernel='linear')
    y2_train = y2[train]

    clf.fit(GEN_Anat100, y2_train)
    prediction_Anat = clf.predict(X1_test[:,GEN_Anat100_Index])
    result = prediction_Anat == y2_test
    accuracy_GEN_Anat[i] = sum(result)/result.shape
    Predict_GEN_Anat.extend(list(prediction_Anat))

    clf.fit(GEN_Func100, y2_train)
    prediction_Func = clf.predict(X2_test[:,GEN_Func100_Index])
    result = prediction_Func == y2_test
    accuracy_GEN_Func[i] = sum(result)/result.shape
    Predict_GEN_Func.extend(list(prediction_Func))

    clf.fit(GEN_200, y2_train)
    GEN_200_test = np.concatenate((X1_test[:,GEN_Anat100_Index],X2_test[:,GEN_Func100_Index]),axis=1)
    prediction_Both = clf.predict(GEN_200_test)
    result = prediction_Both == y2_test
    accuracy_GEN_Both[i] = sum(result)/result.shape
    Predict_GEN_Both.extend(list(prediction_Both))

    GEN_True.extend(list(y2_test))

    i = i + 1

# confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(GEN_True,Predict_GEN_Anat,labels=["SCZ(&SIB)","CON(&SIB)"])
confusion_matrix(GEN_True,Predict_GEN_Func,labels=["SCZ(&SIB)","CON(&SIB)"])
confusion_matrix(GEN_True,Predict_GEN_Both,labels=["SCZ(&SIB)","CON(&SIB)"])

# t-test
stats.ttest_1samp(a=accuracy_GEN_Anat,popmean=0.5)
stats.ttest_1samp(a=accuracy_GEN_Func,popmean=0.5)
stats.ttest_1samp(a=accuracy_GEN_Both,popmean=0.5)

stats.ttest_rel(accuracy_GEN_Anat,accuracy_GEN_Func)
stats.ttest_rel(accuracy_GEN_Anat,accuracy_GEN_Both)
stats.ttest_rel(accuracy_GEN_Func,accuracy_GEN_Both)

# accuracy report
accuracy_GEN_Anat_mean = accuracy_GEN_Anat.mean()
accuracy_GEN_Anat_std = accuracy_GEN_Anat.std()
print("%.2f" % accuracy_GEN_Anat_mean + " " + u"\u00B1" + " " + "%.3f" % accuracy_GEN_Anat_std)

accuracy_GEN_Func_mean = accuracy_GEN_Func.mean()
accuracy_GEN_Func_std = accuracy_GEN_Func.std()
print("%.2f" % accuracy_GEN_Func_mean + " " + u"\u00B1" + " " + "%.3f" % accuracy_GEN_Func_std)

accuracy_GEN_Both_mean = accuracy_GEN_Both.mean()
accuracy_GEN_Both_std = accuracy_GEN_Both.std()
print("%.2f" % accuracy_GEN_Both_mean + " " + u"\u00B1" + " " + "%.3f" % accuracy_GEN_Both_std)

np.save("processdata/Anat100index_GEN",Anat100index_GEN)
np.save("processdata/Func100index_GEN",Func100index_GEN)

# get 15 anat
(unique, counts) = np.unique(Anat100index_GEN,return_counts=True)
candidates = unique[np.logical_or(counts == 9,counts == 10)]
rank = np.zeros((len(candidates)))
for i in range(len(candidates)):
    rank[i] = np.where(Anat100index_GEN == candidates[i])[1].mean()
Anat15Index_GEN = candidates[rank.argsort()][0:15]

# get 15 func
(unique, counts) = np.unique(Func100index_GEN,return_counts=True)
candidates = unique[np.logical_or(counts == 9,counts == 10)]
rank = np.zeros((len(candidates)))
for i in range(len(candidates)):
    rank[i] = np.where(Func100index_GEN == candidates[i])[1].mean()
Func15Index_GEN = candidates[rank.argsort()][0:15]

OriginFuncIndex_GEN = np.zeros((15,2))
for k in range(15):
    for i in range(1, 183):
        for j in range(0, i):
            index = (i-1)*i/2 + j
            if index == Func15Index_GEN[k]:
                OriginFuncIndex_GEN[k,0] = i + 1
                OriginFuncIndex_GEN[k,1] = j + 1
UniqueFuncIndex_GEN = np.unique(OriginFuncIndex_GEN)

# into csv
SCZSIB_15CT = np.zeros((15))
CONSIB_15CT = np.zeros((15))
CortThickness = np.load("processdata/CortThickness.npy")

for k in range(15):
    SCZSIB_15CT[k] = CortThickness[y2 == "SCZ(&SIB)",int(Anat15Index_GEN[k])].mean()
    CONSIB_15CT[k] = CortThickness[y2 == "CON(&SIB)", int(Anat15Index_GEN[k])].mean()

pd.DataFrame(SCZSIB_15CT).to_csv("processdata/SCZ&SIB_15CT.csv")
pd.DataFrame(CONSIB_15CT).to_csv("processdata/CON&SIB_15CT.csv")

SCZSIB_15FC = np.zeros((15))
CONSIB_15FC = np.zeros((15))
FuncConnectivity = np.load("FuncConnectivity.npy")
FuncConnectivity = np.mean(FuncConnectivity,axis=1)

for k in range(15):
    SCZSIB_15FC[k] = FuncConnectivity[y2 == "SCZ(&SIB)",int(OriginFuncIndex_GEN[k,0] -1),int(OriginFuncIndex_GEN[k,1] -1)].mean()
    CONSIB_15FC[k] = FuncConnectivity[y2 == "CON(&SIB)",int(OriginFuncIndex_GEN[k,0] -1),int(OriginFuncIndex_GEN[k,1] -1)].mean()

pd.DataFrame(SCZSIB_15FC).to_csv("processdata/SCZSIB_15FC.csv")
pd.DataFrame(CONSIB_15FC).to_csv("processdata/CONSIB_15FC.csv")

# plot 15 anat
parcel_file = 'rois/HCP-MMP_1mm.nii.gz'
Glasser_180 = nimg.load_img(parcel_file)

for k in range(len(Anat15Index_GEN)):
    if Anat15Index_GEN[k] > 180:
        Anat15Index_GEN[k] = Anat15Index_GEN[k] + 20

Final_Anat_GEN = nimg.math_img('0*a', a=Glasser_180)

for i in range(len(Anat15Index_GEN)):
    Formula1 = 'a == ' + str(Anat15Index_GEN[i])
    roi_mask = nimg.math_img(Formula1, a=Glasser_180)
    Formula2 = 'a + (' + str(round(SCZSIB_15CT[i],3)) + ' * b)'
    Final_Anat_GEN = nimg.math_img(Formula2, a=Final_Anat_GEN, b=roi_mask)

plotting.plot_glass_brain(Final_Anat_GEN)
plt.savefig("results/GEN.png")