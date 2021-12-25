from nilearn import input_data
from nilearn import image as nimg

# Load parcellation file
parcel_file = 'G:/Python/resources/rois/RelabeledGlasser2016.nii.gz'
Glasser_180 = nimg.load_img(parcel_file)

masker = input_data.NiftiLabelsMasker(labels_img=Glasser_180,
                                      standardize=True,
                                      detrend=True,
                                      low_pass=0.08,
                                      high_pass= 0.009,
                                      t_r=2.5)

