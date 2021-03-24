import nibabel
import os.path as op 
import numpy as np 

def load_basil(from_dir, pvec=True):
    if pvec: 
        gm = op.join(from_dir, 'basil_out', 'step2', 'mean_ftiss.nii.gz')
        wm = op.join(from_dir, 'basil_out', 'step2', 'mean_fwm.nii.gz')
        return np.stack([
            nibabel.load(p).get_fdata() for p in [gm, wm]
        ], axis=-1)
    
    else: 
        gm = op.join(from_dir, 'basil_out', 'step1', 'mean_ftiss.nii.gz')
        return nibabel.load(gm).get_fdata()
    
def load_oxasl(from_dir, pvec=True):
    if pvec: 
        gm = op.join(from_dir, 'output_pvcorr', 'native', 'perfusion_calib.nii.gz')
        wm = op.join(from_dir, 'output_pvcorr', 'native', 'perfusion_wm_calib.nii.gz')
        return np.stack([
            nibabel.load(p).get_fdata() for p in [gm, wm]
        ], axis=-1)
    
    else: 
        gm = op.join(from_dir, 'output', 'native', 'perfusion.nii.gz')
        return nibabel.load(gm).get_fdata()
    
def load_lr(from_dir):
    gm = op.join(from_dir, 'gm', 'basil_out', 'step1', 'mean_ftiss.nii.gz')
    wm = op.join(from_dir, 'wm', 'basil_out', 'step1', 'mean_ftiss.nii.gz')
    return np.stack([
        nibabel.load(p).get_fdata() for p in [gm, wm]
    ], axis=-1)

def load_oxasl_lr(from_dir, pvec=True):
    gm = op.join(from_dir, 'gm', 'output', 'native', 'perfusion_calib.nii.gz')
    wm = op.join(from_dir, 'wm', 'output', 'native', 'perfusion_calib.nii.gz')
    return np.stack([
        nibabel.load(p).get_fdata() for p in [gm, wm]
    ], axis=-1)