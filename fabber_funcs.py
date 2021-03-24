
import numpy as np 
import os.path as op 
import regtricks as rt 
import os 
import nibabel

def basil_cmd(asl, mask, opts, odir, pvs=None):
    
    os.makedirs(odir, exist_ok=True)
    spc = rt.ImageSpace(mask)

    if pvs is not None: 
        gpath = op.join(odir, 'pvgm.nii.gz')
        wpath = op.join(odir, 'pvwm.nii.gz')
        pvs = nibabel.load(pvs).get_fdata()
        for path,arr in zip([gpath,wpath], [pvs[...,0],pvs[...,1]]):
            spc.save_image(arr,path)

    optsfile = op.join(odir, 'basil_opts.txt')
    with open(optsfile, 'w') as f: 

        for key,val in opts.items():
            if val is not True: 
                f.write(f"--{key}={val}\n")
            else: 
                f.write(f"--{key}\n")

        f.write("\n")

    odir = op.join(odir, 'basil_out')
    cmd = f"basil -i {asl} -o {odir} -m {mask}"
    cmd += f" --optfile={optsfile} "
    if pvs is not None: 
        cmd += f" --pgm={gpath} --pwm={wpath}"

    return cmd 

    # # if no PVEc, return a single volume of CBF
    # # if PVEc, return a 4D volume, arranged (GM,WM) in last dimension
    # if pvcorr: 
    #     gcbf = nibabel.load(op.join(odir, 
    #                     'step2/mean_ftiss.nii.gz'))
    #     wcbf = nibabel.load(op.join(odir, 
    #                     'step2/mean_fwm.nii.gz'))
    #     cbfs = np.stack((gcbf.get_fdata(), wcbf.get_fdata()), axis=-1)
    #     gatt = nibabel.load(op.join(odir, 
    #                     'step2/mean_delttiss.nii.gz'))
    #     watt = nibabel.load(op.join(odir, 
    #                     'step2/mean_deltwm.nii.gz'))
    #     atts = np.stack((gatt.get_fdata(), watt.get_fdata()), axis=-1)

    #     log = open(op.join(odir, 'step2/logfile'), 'r').read()
    #     idx = log.index('*** Spatial iteration *** 200')
    #     log = log[idx:]
    #     substr = 'SpatialPrior::Calculate aK 0: New aK: '
    #     start = log.index(substr) + len(substr)
    #     end = log[start:].index('\n')
    #     final_ak = float(log[start:start+end])
    #     print('final ak', final_ak)

    # else: 
    #     cbfs = nibabel.load(op.join(odir, 
    #                     'step1/mean_ftiss.nii.gz')).get_fdata()
    #     atts = nibabel.load(op.join(odir, 
    #                     'step1/mean_delttiss.nii.gz')).get_fdata()
        
    # if odir is None: 
    #     d.clean_up()
        
    # return cbfs, atts



def oxasl_cmd(asl, calib, mask, odir, opts, pvs=None):
    

    spc = rt.ImageSpace(asl)
    os.makedirs(odir, exist_ok=True)

    optspath = op.join(odir, 'bopts.txt')
    with open(optspath, 'w') as f: 
        for key,val in opts.items():
            if val is not True: 
                f.write(f"--{key}={val}\n")
            else: 
                f.write(f"--{key}\n")

        f.write("\n")

    cmd = [
        "oxasl", "-i", asl,
        "-c", calib, "-m", mask, "--calib-aslreg",
        "-o", odir, "--tau", "0.7", "--tis", "1.8", "--t1b", "2.1",
        "--tr", "2.861", "--alpha", "0.95", "--debug ",
        "--fa", "70", "--cmethod", "voxel", "--overwrite", "--fixbat",
        "--iaf", "tc", "--te", "14", "--basil-options", optspath,
    ] 

    if pvs is not None: 
        cmd += ["--pvcorr", "--pvgm", pvs[0], "--pvwm", pvs[1],]

    return " ".join(cmd)
#     res = subprocess.run(cmd, stderr=subprocess.PIPE)

#     if pvs is not None: 
#         pvdir = op.join(odir, 'output_pvcorr/native')
#         gcbf = nibabel.load(op.join(pvdir, 
#                         'perfusion_calib.nii.gz'))
#         wcbf = nibabel.load(op.join(pvdir, 
#                         'perfusion_wm_calib.nii.gz'))
#         cbfs = np.stack((gcbf.get_fdata(), wcbf.get_fdata()), axis=-1)

#     else: 
#         pvdir = op.join(odir, 'output/native')
#         cbfs = nibabel.load(op.join(pvdir, 
#                         'perfusion_calib.nii.gz')).get_fdata()

#     return cbfs
