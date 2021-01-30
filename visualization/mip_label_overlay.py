import os
import re
import numpy as np

from prep import RsomVisualization
from utils import get_unique_filepath        

def mip_label_overlay(file_ids, dirs, plot_epidermis=False):

    mat_dir = dirs['in']
    seg_dir_lay = dirs['layer']
    seg_dir_ves = dirs['vessel']
    out_dir = dirs['out']

    if isinstance(file_ids, str):
        
        file_ids = [file_ids]

    elif file_ids is None:
        
        # get file_ids from vesselseg directory
        file_ids = os.listdir(seg_dir_ves)
        file_ids = [id_[:id_.find('_', 2)] for id_ in file_ids]

    for file_id in file_ids:
        mip_label_overlay1(file_id, 
                           dirs, 
                           plot_epidermis=plot_epidermis, 
                           axis=-1, 
                           return_img=False)

def mip_label_overlay1(file_id, dirs, plot_epidermis=False, axis=-1, return_img=False):
    """ 
    axis=-1 means all axes
    """

    mat_dir = dirs['in']
    seg_dir_lay = dirs['layer']
    seg_dir_ves = dirs['vessel']
    out_dir = dirs['out']

    matLF, matHF = get_unique_filepath(mat_dir, file_id)
    
    _, matLF_ = os.path.split(matLF)
    idx_1 = matLF_.find('_')
    idx_2 = matLF_.find('_', idx_1+1)
    matSurf = os.path.join(mat_dir, 'Surf' + matLF_[idx_1:idx_2+1] + '.mat')
    
    sample = RsomVisualization(matLF, matHF, matSurf)
    sample.read_matlab()
    sample.flat_surface()
    
    # z=500
    sample.cut_depth()
    
    seg_file_ves = get_unique_filepath(seg_dir_ves, file_id)
    seg_file_lay = get_unique_filepath(seg_dir_lay, file_id)
    z0 = int(re.search('(?<=_z)\d{1,3}(?=_)', seg_file_ves).group())
    # print('z0 = ', z0)
    
    if axis == -1 or axis == 0:
        # axis = 0
        # this is the top view
        axis_ = 0
        sample.calc_mip(axis=axis_, do_plot=False, cut_z=z0)
        sample.calc_mip_ves_seg(seg=seg_file_ves, axis=axis_, padding=(0, 0))
        sample.merge_mip_ves(do_plot=False)
        if return_img:
            return sample.return_mip()
        else:
            sample.save_comb_mip(out_dir)
    if axis == -1 or axis == 1:
        # axis = 1
        axis_ = 1
        sample.calc_mip(axis=axis_, do_plot=False, cut_z=0)
        sample.calc_mip_ves_seg(seg=seg_file_ves, axis=axis_, padding=(z0, 0))
        sample.merge_mip_ves(do_plot=False)
        if plot_epidermis:
            sample.calc_mip_lay_seg(seg=seg_file_lay, axis=axis_, padding=(0, 0))
            sample.merge_mip_lay(do_plot=False)
        if return_img:
            return sample.return_mip()
        else:
            sample.save_comb_mip(out_dir)
    
    if axis == -1 or axis == 2:
        axis_ = 2
        sample.calc_mip(axis=axis_, do_plot=False, cut_z=0)
        sample.calc_mip_ves_seg(seg=seg_file_ves, axis=axis_, padding=(z0, 0))
        sample.merge_mip_ves(do_plot=False)
        if plot_epidermis:
            sample.calc_mip_lay_seg(seg=seg_file_lay, axis=axis_, padding=(0, 0))
            sample.merge_mip_lay(do_plot=False)
        if return_img:
            return sample.return_mip()
        else:
            sample.save_comb_mip(out_dir)
