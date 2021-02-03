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

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

sys.path.append('/Users/tom/modules/svb_module')
from svb.main import run 



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


def run_svb(data_nii, mask, opts, pvs=None):

    svb_opts = {
        "learning_rate" : 0.02,
        "batch_size" : 5,
        "sample_size" : 5,
        "epochs" : 2500,
        "prior_type": "M",
        "ak": 1e-1, 
        "display_step": 100,
        "infer_ak": True, 
        "save_mean": True, 
        "save_history": True, 
        "log_stream" : sys.stdout,
        
        "casl": True, 
        "tau": MODEL_OPTIONS["tau"],
        "alpha": MODEL_OPTIONS["alpha"], 
        "lambda": MODEL_OPTIONS["lambda"], 
        "lambdawm": MODEL_OPTIONS["lambdawm"],
        "t1": MODEL_OPTIONS["t1"],
        "t1b": MODEL_OPTIONS["t1b"],
        "t1wm": MODEL_OPTIONS["t1wm"],
        "att": PARAMS["GM_BAT"],
        "attwm": PARAMS["WM_BAT"],
        "attsd": MODEL_OPTIONS["batsd"],
    }
    
    svb_opts.update({
        "mask" : mask,
        "plds": [ MODEL_OPTIONS[f'pld{i}'] for i in range(1,6) ], 
        "repeats": MODEL_OPTIONS['repeats'], 
    })
    
    if pvs is not None: 
        svb_opts.update({
            "pvcorr" : True,
            "pvgm" : pvs[...,0],
            "pvwm": pvs[...,1],
        })

    runtime, svb, training_history = run(
        data_nii, "aslrest",
        "svb_out", 
        **svb_opts)
    
    means = svb.evaluate(svb.model_means)
    if pvs is not None: 
        cbf = np.zeros((*mask.shape, 2))
        cbf[mask,0] = means[0]
        cbf[mask,1] = means[2]
    else: 
        cbf = np.zeros(mask.shape)
        cbf[mask] = means[0]
        
    return cbf
    
def run_basil(data, mask, opts, pvs=None):
    
#     with tempfile.TemporaryDirectory() as d: 
        d = 'tempdir'
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
       
            f.write("--casl\n--fixbolus\n--fixbat\n")
            f.write(f"--bat={PARAMS['GM_BAT']}\n--batwm={PARAMS['WM_BAT']}\n")
            for key in ['lambda', 'lambdawm', 't1', 'alpha', 'tau', 't1b', 't1wm']:
                 f.write(f"--{key}={opts[key]}\n")
                    
            for n in range(1, 5+1):
                pld = opts[f'pld{n}']
                f.write(f"--pld{n}={pld}\n")
                f.write(f"--rpt{n}={opts['repeats']}\n")
   
            f.write("\n\n")
            
        odir = op.join(d, 'basil_out')
        cmd = f"basil -i {dpath} -o {odir} -m {mpath}"
        cmd += f" --optfile={optsfile} "
        if pvcorr: 
              cmd += f" --pgm={gpath} --pwm={wpath}"
               
        # print the oxasl cmd, and the extra basil options 
#         print(cmd)
#         print("basil options:", 
#               open('tempdir/basil_opts.txt', 'r').read().replace('\n', ' '), '\n\n')
        subprocess.run(cmd, shell=True, check=True)

        # if no PVEc, return a single volume of CBF
        # if PVEc, return a 4D volume, arranged (GM,WM) in last dimension
        if pvcorr: 
            gcbf = nibabel.load(op.join(odir, 
                            'step2/mean_ftiss.nii.gz'))
            wcbf = nibabel.load(op.join(odir, 
                            'step2/mean_fwm.nii.gz'))
            ret = np.stack((gcbf.get_fdata(), wcbf.get_fdata()), axis=-1)

        else: 
            cbf = nibabel.load(op.join(odir, 
                            'step1/mean_ftiss.nii.gz')).get_fdata()
            ret = cbf
        return ret 


# def run_basil(data, mask, pvgm, pvwm):
    
# #     with tempfile.TemporaryDirectory() as d: 
#         d = op.abspath('tempdir')
#         os.makedirs(d, exist_ok=True)
#         opts = "\n".join([f'--{k}={v}' 
#                                for (k,v) in BASIL_OPTIONS.items()])
# #         opts += "\n--casl"
#         optsfile = op.join(d, 'basilopts.txt')
#         with open(optsfile, 'w') as f: 
#             f.write(opts)
            
#         spc = rt.ImageSpace.create_axis_aligned([0,0,0], mask.shape, [1,1,1])
#         dpath = op.join(d, 'data.nii.gz')
#         mpath = op.join(d, 'mask.nii.gz')
#         gpath = op.join(d, 'pvgm.nii.gz')
#         wpath = op.join(d, 'pvwm.nii.gz')
#         for path,arr in zip([dpath,mpath,gpath,wpath],
#                             [data,mask,pvgm,pvwm]):
#             spc.save_image(arr,path)
        
#         odir = op.join(d, 'basil_out')
#         cmd = ("/usr/local/fsl/bin/basil "
#               + f"-i {dpath} -o {odir} -m {mpath} "
#               + f"--optfile={optsfile} --pgm={gpath} --pwm={wpath}") 
        
#         print(cmd)
#         subprocess.run(cmd, shell=True)
        
#         return nibabel.load(op.join(odir, 'step1/mean_ftiss.nii.gz')).get_fdata()
#         cbfg = nibabel.load(op.join(odir, 'step2/mean_ftiss.nii.gz')).get_fdata()
#         cbfw = nibabel.load(op.join(odir, 'step2/mean_fwm.nii.gz')).get_fdata()
#         return np.stack((cbfg, cbfw), axis=-1)


# def run_oxasl(data, mask, opts, pvs=None):
    
# #     with tempfile.TemporaryDirectory() as d: 
#         d = 'tempdir'
#         os.makedirs(d, exist_ok=True)

#         # PVEc is enabled if pvs are provided 
#         pvcorr = (pvs is not None)
    
#         # dump all arrays to nifti in the working directory 
#         spc = rt.ImageSpace.create_axis_aligned([0,0,0], data.shape[:3], [1,1,1])
#         if pvcorr: 
#             gpath = op.join(d, 'pvgm.nii.gz')
#             wpath = op.join(d, 'pvwm.nii.gz')
#             for path,arr in zip([gpath,wpath], [pvs[...,0],pvs[...,1]]):
#                 spc.save_image(arr,path)

#         dpath = op.join(d, 'data.nii.gz')
#         mpath = op.join(d, 'mask.nii.gz')
#         for path,arr in zip([dpath,mpath],[data,mask]):
#             spc.save_image(arr,path)
                
                
#         # extra options for BASIL 
#         # if BAT are present in opts dict, then we will set --fixbat later on 
#         optsfile = op.join(d, 'basil_opts.txt')
#         with open(optsfile, 'w') as f: 
#             f.write("\n".join([f"lambda={opts['lambda']}", 
#                                f"lambdawm={opts['lambdawm']}",
#                                 "--t1wm=1.3"]))
#             if 'WM_BAT' in opts: 
#                 f.write("\n".join(["",
#                                f"batwm={opts['WM_BAT']}", 
#                                f"bat={opts['GM_BAT']}"]))
#             f.write("\n")
            
#         pldstr = " --plds=%s" % (",".join([ str(opts[f'pld{n}']) for n in range(1,5+1)]))
#         odir = op.join(d, 'oxasl_out')
#         cmd = ("oxasl --iaf=diff --overwrite --no-report --artoff"
#               + f" --fixbolus --ibf=tis --basil-options={optsfile}"
#               + f" --t1={opts['t1']} --t1b={opts['t1b']}" 
#               + f" --alpha={opts['alpha']} --casl --rpts={opts['repeats']}" 
#               + f" -i {dpath} -o {odir} -m {mpath} --debug "
#               + f" --tau={opts['tau']} {pldstr} --batsd={opts['batsd']}")
#         if pvcorr: 
#               cmd += f" --pvcorr --pvgm={gpath} --pvwm={wpath}"
#         if 'WM_BAT' in opts: 
#             cmd += " --fixbat"

#         # print the oxasl cmd, and the extra basil options 
#         print(cmd)
#         print("basil options:", 
#               open('tempdir/basil_opts.txt', 'r').read().replace('\n', ' '), '\n\n')
#         subprocess.run(cmd, shell=True, check=True)

#         # if no PVEc, return a single volume of CBF
#         # if PVEc, return a 4D volume, arranged (GM,WM) in last dimension
#         if pvcorr: 
#             gcbf = nibabel.load(op.join(odir, 
#                             'output_pvcorr/native/perfusion.nii.gz'))
#             wcbf = nibabel.load(op.join(odir, 
#                             'output_pvcorr/native/perfusion_wm.nii.gz'))
#             ret = np.stack((gcbf.get_fdata(), wcbf.get_fdata()), axis=-1)

#         else: 
#             cbf = nibabel.load(op.join(odir, 
#                             'output/native/perfusion.nii.gz')).get_fdata()
#             ret = cbf
#         return ret 