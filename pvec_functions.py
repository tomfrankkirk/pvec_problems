import numpy as np 
from scipy import sparse
import scipy.linalg as spla 
import copy
import regtricks as rt 
from pdb import set_trace


def adjacency_matrix(xy_shape):
    # Construct the connectivity matrix of the first row: zeros all the way along, except for the first neighbour
    row = np.zeros(xy_shape[0])
    row[1] = 1 

    # The Toeplitz matrix for this represents the connectivity of the elements within the first row to the 
    # other elements within the row
    sub = spla.toeplitz(row)

    # We want to repeat this pattern for all the other rows: we form a bigger matrix, where the Toeplitz submatrix
    # is repeated along the main diagonal. This represents the linear offset required to index items in the Nth row 
    adj = np.kron(np.eye(xy_shape[1]),sub)

    # For column connectivity, element n connects to element n-N and n+N. These are two big diagonals, offset from 
    # the main by N elements. We also add the diagonal neighbours, ie, n-N+1, n-N-1, n+N+1, n+N-1. These sit as parallel 
    # and adjacent diagonals on either side of the preceeding diagonals  
    # The Kroenecker matrix evaluated the other way round gets us this
    col = np.eye(xy_shape[1])
    col += np.diag(np.ones(col.shape[0]-1),1)
    col += np.diag(np.ones(col.shape[0]-1),-1)
    adj += np.kron(sub,col)

    # Finally, for convenience set the diagonal to 1 (easier to form kernels when doing LR PVEc)
    adj = sparse.dok_matrix(adj, dtype=bool)
    adj[np.diag_indices(adj.shape[0])] = 1
    return adj.tocsr()

def pvec_lr(cbf_pv, pvs, mask, slice_no):
    
    adj = adjacency_matrix(cbf_pv.shape[:2])
    
    if pvs.shape[3] > 2:
        pvs = pvs[...,:2]

    cbf_pv = cbf_pv[...,slice_no]
    pvs = pvs[...,slice_no,:]
    mask = mask[...,slice_no]
    
    cbf_corrected = np.zeros((mask.sum(), 2))
    pvs = pvs.reshape(-1,2)
    cbf_pv = cbf_pv.flatten()
    assert mask.size == pvs.shape[0]

    for idx,vidx in enumerate(np.flatnonzero(mask)):
        neighbours = adj[vidx,:].indices
        y = cbf_pv[neighbours]
        w = pvs[neighbours,:]
        cbf = np.linalg.lstsq(w, y)[0]
        cbf_corrected[idx,:] = cbf

    return cbf_corrected

def rescale_pvs(pvs):
    pvs = copy.deepcopy(pvs)
    sums = pvs.sum(-1)
    mask = (sums > 0)
    pvs[mask,:] = pvs[mask,:] / sums[mask,None]
    assert np.abs(pvs[mask,:].sum(-1) - 1).max() < 1e-3 
    return pvs 

def calc_sphere_pvs(cent, r_g, r_w, spc, superfactor):
    spc_high = spc.resize_voxels(1/superfactor)
    xyz = spc_high.voxel_centres()[:,:,0,:]
    r = np.sqrt(((xyz[:,:,0] - cent[0]) ** 2) + ((xyz[:,:,1] - cent[1]) ** 2))
    wm = (r < r_g) & (r < r_w)
    gm = (r < r_g) & (r >= r_w)
    wm_pv = rt.application_helpers.sum_array_blocks(wm[:,:,None], [superfactor,superfactor,1]) / (superfactor ** 2)
    gm_pv = rt.application_helpers.sum_array_blocks(gm[:,:,None], [superfactor,superfactor,1]) / (superfactor ** 2)
    pvs = np.stack([gm_pv, wm_pv], axis=-1)
    return pvs 



def super_source_resample(data, transform, src_spc, ref_spc, superfactor):
    
    factor = np.array(superfactor)
    
    # Create super-resolution voxel grid for source, and interpolate
    # source data onto it 
    super_source_spc = src_spc.resize_voxels(1 / factor)
    super_source_data = rt.Registration.identity().apply_to_array(
            data, src_spc, super_source_spc, superfactor=False)
    
    # Generate the ijk grid of voxel centres for the super grid, push 
    # them into the reference voxel space, and convert into flat voxel indices
    # denoting which reference voxel they are contained within. 
    super_ijk = super_source_spc.voxel_centres().reshape(-1,3)
    super_ijk_inref = rt.application_helpers.aff_trans(
        transform.src2ref, super_ijk)
    super_ijk_inref_vox = rt.application_helpers.aff_trans(
        ref_spc.world2vox, super_ijk_inref)
    super_ijk_inref_vox = np.round(super_ijk_inref_vox).astype(np.int32)
    
    # Create a FoV mask to catch voxels that have been pushed outside the 
    # reference FoV - we ignore these. 
    fov_mask = ((super_ijk_inref_vox >= 0) 
                & (super_ijk_inref_vox < ref_spc.size)).all(-1)
    super_ijk_inref_vox = super_ijk_inref_vox[fov_mask,:]
    super_ijk_inref_voxidx = np.ravel_multi_index(super_ijk_inref_vox.T, 
                                                  ref_spc.size)

    # Use bincount to count how many super-resolution voxels are contained
    # within each reference voxel, and use the weights argument to sum 
    # corresponding voxel values from the super-resolution source data 
    super_source_data = super_source_data.flatten()[fov_mask]
    out_super = np.bincount(super_ijk_inref_voxidx, 
                        weights=super_source_data, 
                        minlength=super_ijk_inref_voxidx.max())
    
    # Divide by number of super voxels in each ref voxel to get an average. 
    counts = np.bincount(super_ijk_inref_voxidx)
    nonzero = (counts > 0)
    out_super[nonzero] /= counts[nonzero]
    
    # Reshape back to ref size. 
    return out_super.reshape(ref_spc.size)


def super_ref_resample(data, transform, src_spc, ref_spc, superfactor):
    
    factor = np.array(superfactor)
    super_ref_spc = ref_spc.resize_voxels(1 / factor)
    super_out = transform.apply_to_array(data, src_spc, 
                                         super_ref_spc, superfactor=False)
    out = rt.application_helpers.sum_array_blocks(super_out, factor)
    return out / np.prod(factor)
