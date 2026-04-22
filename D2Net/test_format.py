import nibabel as nib

img = nib.nifti1.load("/work/grana_neuro/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/BraTS-GLI-00021-000/BraTS-GLI-00021-000-t2f.nii.gz").get_fdata()
print(type(img))
print(img.shape)