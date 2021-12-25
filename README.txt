### 360 Cortical Thickness

1. GetCortThick.py: combine cortical thickness data from '/thickness' (which is from Freesurfer on brainlife)

	return CortThickness.npy

### 16653 Functional Connectivity

1. GetROI.py: get network ROIs

	return RelabeledGlasser2016.nii.gz

2. GetFC.py: get all the functional connectivity
	
	return FuncConnectivity.npy

### combine Regressor

1. GetRegressor

	return Reg_z.npy

### SVM

1. ClassifierHEL.py

2. ClassifierGEN.py
