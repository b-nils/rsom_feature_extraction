import numpy as np
import warnings

def get_patches(volume, divs, offset):
    '''
    Args:
        - volume (np.array)         :   The volume to cut
                                        N Dimensions:
                                        single channel   : (X_1,..., X_N)
                                        multi channel    : (X_1,..., X_N, C)
        - divs (tuple)              :   Amount to divide each dimension
                                        len(divs) must be equal to N 
        - offset (tuple)            :   Offset for each div
                                        len(offset) must be equal to N
                                      
    Output:
        - patches (np.array)        :   patches stacked along first dimension
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)
    
    assert len(volume.shape) == len(divs) or len(volume.shape) == len(divs) + 1
    assert len(volume.shape) == len(offset) or len(volume.shape) == len(offset) + 1

    patches = []
    # simply iterate over all indices
    for idx in np.arange(np.prod(divs)):
        patches.append(get_patch(volume, idx, divs, offset))
    
    # TODO use stack
    return np.array(patches)

def get_patch(volume, index, divs=(2,2,2), offset=(6,6,6)):
    '''
    Args:
        - volume (np.array)         :   The volume to cut
                                        N Dimensions:
                                        single channel   : (X_1,..., X_N)
                                        multi channel    : (X_1,..., X_N, C)
        - index (int)               :   flattened patch iterator.
                                        in range 0 to prod(divs)-1
        - divs (tuple)              :   Amount to divide each dimension
                                        len(divs) must be equal to N 
        - offset (tuple)            :   Offset for each div
                                        len(offset) must be equal to N
                                        
    Output:
        - patch (np.array)          :   patch at index
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)
    
    assert len(volume.shape) == len(divs) or len(volume.shape) == len(divs) + 1
    assert len(volume.shape) == len(offset) or len(volume.shape) == len(offset) + 1
    
    
    
    if len(volume.shape) == len(divs) + 1:
        # multi channel
        shape = volume.shape[:-1]           
    else:
        # single channel
        shape = volume.shape
        
    if np.any(np.mod(shape, divs)):
        warnings.warn(('At least one dimension of the input volume can\'t be '
                       'divided by divs without remainder. Your input shape '
                       'and reconstructed shapes won\'t match.'))
        
    widths = [int(s/d) for s, d in zip(shape, divs)]
    patch_shape = [w+o*2 for w, o in zip(widths, offset)]
    
    # create nd index
    index_ = np.unravel_index(index, divs)
    
    # coordinates
    c = [s*d for s, d in zip(index_, widths)] 
    
    if len(volume.shape) == len(divs) + 1:
        patch_shape = tuple(patch_shape + [volume.shape[-1]])
    else:
        patch_shape = tuple(patch_shape)
        
    patch = np.zeros(patch_shape, dtype=volume.dtype)
    
    s_ = []
    e_ = []
    slice_idx = []
    slice_idx_patch = []
    # for every dimension X_1 ... X_N
    for dim in np.arange(len(c)):
        # calculate start and end index of the patch
        s_ = c[dim] - offset[dim] if c[dim] - offset[dim] >= 0 else 0
        e_ = c[dim] + widths[dim] + offset[dim] if \
            c[dim] + widths[dim] + offset[dim] <= shape[dim] else shape[dim]
        slice_idx.append(slice(s_, e_))
        
        # start and end index considering offset
        ps_ = offset[dim] - (c[dim] - s_)
        pe_ = ps_ + (e_ - s_)
        slice_idx_patch.append(slice(ps_, pe_))

    slice_idx = tuple(slice_idx)
    slice_idx_patch = tuple(slice_idx_patch)
    
    # cut out current patch
    vp = volume[slice_idx]
    
    # for offset
    patch[slice_idx_patch] = vp
    return patch

def get_volume(patches, divs = (2,2,3), offset=(6,6,6)):
    '''
    Args:
        - patches (np.array)         :  The patches to reconstruct. N_P patches
                                        are stacked along first dimension.
                                        single channel : (N_P, X_1,..., X_N)
                                        multi channel  : (N_P, X_1,..., X_N, C)
        - divs (tuple)              :   Amount to divide each dimension
                                        len(divs) must be equal to N 
        - offset (tuple)            :   Offset for each div
                                        len(offset) must be equal to N
                                      
    Output:
        - volume  (np.array)        :   patches reconstructed to volume
                                        single channel : (X_1,..., X_N)
                                        multi channel  : (X_1,..., X_N, C)
    '''
    if isinstance(divs, int):
        divs = (divs,)
    if isinstance(offset, int):
        offset = (offset,)

    new_shape = [(ps -of*2)*int(d) \
                 for ps, of, d in zip(patches.shape[1:], offset, divs)]
    
    if len(patches.shape) == len(divs) + 2:
        # multi channel
        new_shape = tuple(new_shape + [patches.shape[-1]])
    else:
        # single channel
        new_shape = tuple(new_shape)
    
    volume = np.zeros(new_shape, dtype=patches.dtype)
    shape = volume.shape
    widths = [int(s/d) for s, d in zip(shape, divs)]
    # iterate over patch indices
    for index in np.arange(np.prod(divs)):
        index_ = np.unravel_index(index, divs)
        slice_idx = []
        slice_idx_offs = []
        # iterate over dimension X_1 ... X_N
        for dim in np.arange(len(index_)):
            # calculate start and end index inside volume
            s_ = (index_[dim] * widths[dim])
            e_ = ((index_[dim] + 1) * widths[dim])
            slice_idx.append(slice(s_, e_))
            
            # calculate start and end index inside patch,
            # to ret rid of the offset
            ps_ = offset[dim]
            pe_ = offset[dim] + widths[dim]
            slice_idx_offs.append(slice(ps_, pe_))
            
        patch = patches[index,...]
        volume[tuple(slice_idx)] = patch[tuple(slice_idx_offs)]
    return volume

        


