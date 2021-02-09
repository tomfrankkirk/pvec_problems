from fabber import FabberCl
import fabber 
import numpy as np 
import tempfile
import os.path as op 
import subprocess 
import regtricks as rt 
import os 
import nibabel
import shutil
import sys


from pdb import set_trace

# For the fabber forward model 
N_PLDS = 5
MODEL_OPTIONS = {
    "model": "aslrest", 
    "pld1": 1.25,
    "pld2": 1.5,
    "pld3": 1.75,
    "pld4": 2,
    "pld5": 2.25,
    "tau": 1.0,
    "casl": True,
    "inctiss": True,
    "incbat": True,
    "alpha": 1, 
    "lambda": 0.98, 
    "lambdawm": 0.8,
    "t1": 1.3,
    "t1b": 1.65,
    "batsd": 1e-12,
    "pvcorr": True,
    "t1wm": 1.3,
    "repeats": 5
}

PARAMS = {
    'GM_CBF': 60, 'WM_CBF': 20,
    'GM_BAT': 1.6, 'WM_BAT': 1.6,
}


# Simulate data using fabber's forward model 
def get_fabber_data(cbf, bat, pvs, opts): 
    fab = FabberCl()
    nvols = N_PLDS * opts['repeats']
    
    gm = fab.model_evaluate(opts, {
        'ftiss': cbf[0], 
        'delttiss': bat[0],
        'fwm': cbf[1],
        'deltwm': bat[1],
        'pvgm': 1.0, 
        'pvwm': 0.0,
    }, nvols)
    
    wm = fab.model_evaluate(opts, {
        'ftiss': cbf[0], 
        'delttiss': bat[0],
        'fwm': cbf[1],
        'deltwm': bat[1],
        'pvgm': 0.0, 
        'pvwm': 1.0,
    }, nvols)
    
    ones = np.ones((*pvs.shape[:3], nvols))
    out = ((ones * np.array(gm)[None,None,None,:] * pvs[...,0,None])
          + (ones * np.array(wm)[None,None,None,:] * pvs[...,1,None]))
        
    return out 



def run_basil(data, mask, opts, pvs=None):
    
#     with tempfile.TemporaryDirectory() as d: 
        d = 'tempdir'
        os.makedirs(d, exist_ok=True)
        shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

        # PVEc is enabled if pvs are provided 
        pvcorr = (pvs is not None)
    
        # dump all arrays to nifti in the working directory 
        spc = rt.ImageSpace.create_axis_aligned([0,0,0], data.shape[:3], [1,1,1])
        if pvcorr: 
            gpath = op.join(d, 'pvgm.nii.gz')
            wpath = op.join(d, 'pvwm.nii.gz')
            for path,arr in zip([gpath,wpath], [pvs[...,0],pvs[...,1]]):
                spc.save_image(arr,path)

        dpath = op.join(d, 'data.nii.gz')
        mpath = op.join(d, 'mask.nii.gz')
        for path,arr in zip([dpath,mpath],[data,mask]):
            spc.save_image(arr,path)
                
        optsfile = op.join(d, 'basil_opts.txt')
        with open(optsfile, 'w') as f: 
       
            for key,val in opts.items():
                if val is not True: 
                    f.write(f"--{key}={val}\n")
                else: 
                    f.write(f"--{key}\n")
                    
            f.write("\n")
            
        odir = op.join(d, 'basil_out')
        cmd = f"basil -i {dpath} -o {odir} -m {mpath} --spatial"
        cmd += f" --optfile={optsfile} "
        if pvcorr: 
              cmd += f" --pgm={gpath} --pwm={wpath}"
               
        # print the oxasl cmd, and the extra basil options 
        print(cmd)
        print("basil options:", 
              open('tempdir/basil_opts.txt', 'r').read().replace('\n', ' '), '\n\n')
        subprocess.run(cmd, shell=True, check=True)

        # if no PVEc, return a single volume of CBF
        # if PVEc, return a 4D volume, arranged (GM,WM) in last dimension
        if pvcorr: 
            gcbf = nibabel.load(op.join(odir, 
                            'step2/mean_ftiss.nii.gz'))
            wcbf = nibabel.load(op.join(odir, 
                            'step2/mean_fwm.nii.gz'))
            cbfs = np.stack((gcbf.get_fdata(), wcbf.get_fdata()), axis=-1)
            gatt = nibabel.load(op.join(odir, 
                            'step2/mean_delttiss.nii.gz'))
            watt = nibabel.load(op.join(odir, 
                            'step2/mean_deltwm.nii.gz'))
            atts = np.stack((gatt.get_fdata(), watt.get_fdata()), axis=-1)

        else: 
            cbfs = nibabel.load(op.join(odir, 
                            'step1/mean_ftiss.nii.gz')).get_fdata()
            atts = nibabel.load(op.join(odir, 
                            'step1/mean_delttiss.nii.gz')).get_fdata()
        return cbfs, atts

