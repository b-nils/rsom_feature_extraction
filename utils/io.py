import os.path
import numpy as np
import nibabel as nib


def save_nii(V, path, fstr=None):
        
    if fstr is not None:
        if not '.nii.gz' in fstr:
            fstr += '.nii.gz'
        path = os.path.join(path, fstr)
        
    V = V.astype(np.uint8)
    img = nib.Nifti1Image(V, np.eye(4))
    nib.save(img, path)

