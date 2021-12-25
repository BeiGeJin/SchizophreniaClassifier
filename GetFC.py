import numpy as np
import pandas as pd

from bids.layout import BIDSLayout
from nilearn import image as nimg
from nilearn.connectome import ConnectivityMeasure

from extractconfounds import extract_confounds
from masker import masker

# Load BIDS
layout = BIDSLayout('fmriprep',
                    config=['bids', 'derivatives'])

# Get all the subjects we have
participants = pd.read_excel('fmriprep/participants.xlsx')
subindex = np.array(participants['subindex'])
condition = participants['Condition3']

# Load parcellation file
parcel_file = 'rois/RelabeledGlasser2016.nii.gz'
Glasser_180 = nimg.load_img(parcel_file)

# Set the list of confound variables we'll be using
confound_variables = ['trans_x', 'trans_y', 'trans_z',
                      'rot_x', 'rot_y', 'rot_z',
                      'global_signal',
                      'white_matter', 'csf']

# Number of TRs we should drop
TR_DROP = 10

# correlation matrix measure
correlation_measure = ConnectivityMeasure(kind='correlation')

# Create space
FuncConnectivity = np.zeros((98,3,183,183))
sub_indexer = 0

# GO!
for sub in subindex:

    # Subject Information
    subinfo = participants.loc[participants['subindex'] == sub]
    subcondit = subinfo.iloc[0]['condit']

    if sub < 10:
        strsub = str(0) + str(sub)
    else:
        strsub = str(sub)

    ###### FC ######
    tasks = ['letter0backtask', 'letter1backtask', 'letter2backtask']
    task_indexer = 0

    for task in tasks:
        func_file = layout.get(subject=strsub,
                               datatype='func',
                               task=task,
                               desc='preproc',
                               space='MNI152NLin2009cAsym',
                               extension="nii.gz",
                               return_type='file')[0]

        confound_file = layout.get(subject=strsub,
                                   datatype='func',
                                   task=task,
                                   desc='confounds',
                                   extension='tsv',
                                   return_type='file')[0]

        # Load the functional file and drop first 4 TR
        func_img = nimg.load_img(func_file)
        func_img = func_img.slicer[:, :, :, TR_DROP:]

        # Extract the confound variables and drop first 4 TR
        confounds = extract_confounds(confound_file,
                                      confound_variables)
        confounds = confounds[TR_DROP:, :]

        # Parcellation + Cleaning + FC matrix
        ROITR = masker.fit_transform(func_img, confounds)
        FC = correlation_measure.fit_transform([ROITR])

        FC = FC[0]
        FuncConnectivity[sub_indexer,task_indexer,:,:] = FC

        task_indexer = task_indexer + 1
    sub_indexer = sub_indexer + 1

np.save("processdata/FuncConnectivity",FuncConnectivity)
