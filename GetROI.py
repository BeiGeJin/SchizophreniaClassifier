import pandas as pd
from nilearn import image as nimg
from nilearn import plotting
import numpy as np
from matplotlib import pyplot as plt

parcel_file = 'rois/HCP-MMP1/HCP-MMP_1mm.nii.gz'
Glasser_180 = nimg.load_img(parcel_file)

ROI_list = pd.read_excel('rois/ROI.xlsx')
ROI_list = list(ROI_list['GlasserIndex'])

atlas_labels = np.unique(Glasser_180.get_fdata())

Final_ROI = nimg.math_img('0*a', a=Glasser_180)

for i in range(len(ROI_list)):
    if ROI_list[i] > 180:
        ROI_list[i] = ROI_list[i] + 20

for i in range(len(ROI_list)):
    Formula1 = 'a == ' + str(ROI_list[i])
    roi_mask = nimg.math_img(Formula1, a=Glasser_180)
    Formula2 = 'a + (' + str(i) + ' +1)*b'
    Final_ROI = nimg.math_img(Formula2, a=Final_ROI, b=roi_mask)

Final_ROI.to_filename('rois/RelabeledGlasser2016.nii.gz')