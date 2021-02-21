"""
Pipeline for cutting epidermis, segmenting vessels and extracting features.

base:
    github: https://github.com/stefanhige/pytorch-rsom-seg
    author: @stefanhige
"""

import os
import numpy as np
from scipy import ndimage
import nibabel as nib
import copy
import torch
import pathlib
from tqdm import tqdm
from skimage import morphology
import multiprocessing
import pandas as pd
import shutil

from prep import Rsom, RsomVessel
from utils import get_unique_filepath
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from laynet import LayerNetBase
from vesnet import VesNetBase
from feature_extraction import feats, feats_utils, pyradiomics, metric_graph
from visualization.mip_label_overlay import mip_label_overlay


def vessel_pipeline(dirs,
                    h_params,
                    device=torch.device('cpu'),
                    divs=(1, 1, 1),
                    pattern=None,
                    delete_tmp=False,
                    n_jobs=15,
                    ):
    """
    Pipeline for RSOM vessel segmentation and feature extraction.
    """
    ###################################################################
    # PREPARING PIPELINE
    ###################################################################
    print('----------------------------------------')
    print('Preparing pipeline ...')
    print('----------------------------------------')

    if dirs is None:
        dirs = {'input': '',
                'output': '',
                'laynet_model': '',
                'vesnet_model': ''}

    # segmentation
    path_tmp_layerseg_prep = os.path.join(dirs['output'], 'tmp', 'layerseg_prep')
    path_tmp_layerseg_out = os.path.join(dirs['output'], 'tmp', 'layerseg_out')
    path_tmp_vesselseg_prep = os.path.join(dirs['output'], 'tmp', 'vesselseg_prep')
    path_tmp_vesselseg_prep_cut = os.path.join(dirs['output'], 'tmp', 'vesselseg_prep_cut')
    path_tmp_vesselseg_out = os.path.join(dirs['output'], 'tmp', 'vesselseg_out')
    path_tmp_vesselseg_out_cut = os.path.join(dirs['output'], 'tmp', 'vesselseg_out_cut')
    path_tmp_vesselseg_out_cut_clean = os.path.join(dirs['output'], 'tmp', 'vesselseg_out_cut_clean')
    path_tmp_vesselseg_prob = os.path.join(dirs['output'], 'tmp', 'vesselseg_out_prob')
    # feature extraction
    path_tmp_ves_sap = os.path.join(dirs['output'], 'tmp', 'ves_sap')
    path_tmp_metric_graph = os.path.join(dirs['output'], 'tmp', 'metric_graph')
    path_tmp_pyradiomics = os.path.join(dirs['output'], 'tmp', 'pyradiomics')
    # visualization
    path_visualization_vessels = os.path.join(dirs['output'], 'visualization', 'segmentation')
    path_visualization_metric_graph = os.path.join(dirs['output'], 'visualization', 'metric_graph')

    for _dir in [os.path.join(dirs['output'], 'tmp'),
                 os.path.join(dirs['output'], 'visualization'),
                 path_tmp_layerseg_prep,
                 path_tmp_layerseg_out,
                 path_tmp_vesselseg_prep,
                 path_tmp_vesselseg_prep_cut,
                 path_tmp_vesselseg_out,
                 path_tmp_vesselseg_out_cut,
                 path_tmp_vesselseg_out_cut_clean,
                 path_tmp_vesselseg_prob,
                 path_tmp_ves_sap,
                 path_tmp_metric_graph,
                 path_tmp_pyradiomics,
                 path_visualization_vessels,
                 path_visualization_metric_graph]:
        if not os.path.isdir(_dir):
            os.mkdir(_dir)
        else:
            print(_dir, 'exists already.')

    num_samples = len([name for name in pathlib.Path(dirs['input']).glob("*LF.mat")])

    # mode
    if pattern is None:
        cwd = os.getcwd()
        os.chdir(dirs['input'])
        all_files = os.listdir()
        os.chdir(cwd)
    else:
        if isinstance(pattern, str):
            pattern = [pattern]
        all_files = [os.path.basename(get_unique_filepath(dirs['input'], pat)[0]) for pat in pattern]

    ###################################################################
    # PREPROCESSING FOR LAYER SEGMENTATION
    ###################################################################
    filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

    print('\n----------------------------------------')
    print('Starting vessel segmentation pipeline ...')
    print('----------------------------------------')

    print('Device is', device)
    print('Files to be processed:')
    for fl in filenameLF_LIST:
        print(fl.replace('LF.mat', ' {LF.mat, HF.mat}'))

    print('\n----------------------------------------')
    print('Preprocessing for epidermis segmentation ...')
    print('----------------------------------------')

    for idx, filenameLF in tqdm(enumerate(filenameLF_LIST), total=num_samples):
        filenameHF = filenameLF.replace('LF.mat', 'HF.mat')

        # extract datetime
        idx_1 = filenameLF.find('_')
        idx_2 = filenameLF.find('_', idx_1 + 1)
        filenameSurf = 'Surf' + filenameLF[idx_1:idx_2 + 1] + '.mat'

        fullpathHF = os.path.join(dirs['input'], filenameHF)
        fullpathLF = os.path.join(dirs['input'], filenameLF)
        fullpathSurf = os.path.join(dirs['input'], filenameSurf)

        Sample = Rsom(fullpathLF, fullpathHF, fullpathSurf)

        Sample.prepare()

        Sample.save_volume(path_tmp_layerseg_prep, fstr='rgb')

    ###################################################################
    # LAYER SEGMENTATION
    ###################################################################
    print('\n----------------------------------------')
    print('Segmenting epidermis ...')
    print('----------------------------------------')

    LayerNetInstance = LayerNetBase(dirs={'model': dirs['laynet_model'],
                                          'pred': path_tmp_layerseg_prep,
                                          'out': path_tmp_layerseg_out},
                                    model_depth=h_params["segmentation"]["laynet_depth"],
                                    device=device)

    LayerNetInstance.predict()

    ###################################################################
    # PREPROCESSING FOR VESSEL SEGMENTATION
    ###################################################################
    print('\n----------------------------------------')
    print('Preprocessing for vessel segmentation ...')
    print('----------------------------------------')

    for idx, filenameLF in tqdm(enumerate(filenameLF_LIST), total=num_samples):
        filenameHF = filenameLF.replace('LF.mat', 'HF.mat')

        # extract datetime
        idx_1 = filenameLF.find('_')
        idx_2 = filenameLF.find('_', idx_1 + 1)
        filenameSurf = 'Surf' + filenameLF[idx_1:idx_2 + 1] + '.mat'

        fullpathHF = os.path.join(dirs['input'], filenameHF)
        fullpathLF = os.path.join(dirs['input'], filenameLF)
        fullpathSurf = os.path.join(dirs['input'], filenameSurf)

        Sample = RsomVessel(fullpathLF, fullpathHF, fullpathSurf)

        Sample.prepare(path_tmp_layerseg_out, mode='pred', fstr='pred.nii.gz')
        Sample.save_volume(path_tmp_vesselseg_prep, fstr='v_rgb')

    ###################################################################
    # VESSEL SEGMENTATION
    ###################################################################
    print('\n----------------------------------------')
    print('Segmenting vessels ...')
    print('----------------------------------------')

    _dirs = {'train': '',
             'eval': '',
             'model': dirs['vesnet_model'],
             'pred': path_tmp_vesselseg_prep,
             'out': path_tmp_vesselseg_out}

    VesNetInstance = VesNetBase(device=device,
                                dirs=_dirs,
                                divs=divs,
                                ves_probability=h_params["segmentation"]["ves_probability"])

    VesNetInstance.predict(use_best=False,
                           save_ppred=True)

    # if save_ppred==True   ^
    # we need to move the probability tensors to another folder
    files = [f for f in os.listdir(path_tmp_vesselseg_out) if 'ppred' in f]

    for f in files:
        shutil.move(os.path.join(path_tmp_vesselseg_out, f),
                    os.path.join(path_tmp_vesselseg_prob, f))

    ###################################################################
    # VESSEL VISUALIZATION
    ###################################################################
    _dirs = {'in': dirs['input'],
             'layer': path_tmp_layerseg_out,
             'vessel': path_tmp_vesselseg_out,
             'out': path_visualization_vessels}

    mip_label_overlay(None, _dirs, roi_z=h_params["post_segmentation"]["roi_z"], plot_epidermis=True)

    ###################################################################
    # POST SEGMENTATION PROCESSING
    ###################################################################
    print('\n----------------------------------------')
    print('Post processing vessel segmentation mask ...')
    print('----------------------------------------')
    image_filenames = pathlib.Path(path_tmp_vesselseg_prep).glob("*_rgb.nii.gz")
    EPIDERMIS_OFFSET = h_params["post_segmentation"]["epidermis_offset"]
    ROI_Z = h_params["post_segmentation"]["roi_z"]
    for i_fn in tqdm(image_filenames, total=num_samples):
        mask_filenames = pathlib.Path(path_tmp_vesselseg_out).glob('*_pred.nii.gz')
        m_fn = [m_fn for m_fn in mask_filenames if "_".join(i_fn.name.split("_")[:3]) in m_fn.name][0]
        fmt = '.nii.gz'
        # cut input image
        img = nib.load(str(i_fn)).get_data()
        img_cut = img[EPIDERMIS_OFFSET:EPIDERMIS_OFFSET+ROI_Z, :, :]
        prefix = i_fn.name.split('.')[0]
        suffix = "_cut"
        img_cut_nifti = nib.Nifti1Image(img_cut, np.eye(4))
        i_cut_fn = prefix + suffix + fmt
        nib.save(img_cut_nifti,  os.path.join(path_tmp_vesselseg_prep_cut, i_cut_fn))
        # cut mask
        mask = nib.load(str(m_fn)).get_data()
        mask_cut = mask[EPIDERMIS_OFFSET:EPIDERMIS_OFFSET+ROI_Z, :, :]
        prefix = m_fn.name.split('.')[0]
        suffix = "_cut"
        mask_cut_nifti = nib.Nifti1Image(mask_cut, np.eye(4))
        m_cut_fn = prefix + suffix + fmt
        nib.save(mask_cut_nifti, os.path.join(path_tmp_vesselseg_out_cut, m_cut_fn))
        # clean mask
        mask_cut_clean = morphology.remove_small_objects(mask_cut.astype(bool),
                                                         h_params["post_segmentation"]["min_size"]).astype(float)
        prefix = m_fn.name.split('.')[0]
        suffix = "_cut_clean"
        mask_cut_clean_nifti = nib.Nifti1Image(mask_cut_clean, np.eye(4))
        m_cut_clean_fn = prefix + suffix + fmt
        nib.save(mask_cut_clean_nifti, os.path.join(path_tmp_vesselseg_out_cut_clean, m_cut_clean_fn))

    ###################################################################
    # VESSAP FEATURE EXTRACTION
    ###################################################################
    print('\n----------------------------------------')
    print('Extrating vesSAP features ...')
    print('----------------------------------------')
    # extract and vesSAP features
    filenames = pathlib.Path(path_tmp_vesselseg_out_cut).glob("*")
    rad_suffix = "_rads"
    fmt = '.nii.gz'
    for fn in tqdm(filenames, total=num_samples):
        fn = str(fn)
        data = feats.preprocess_data(feats_utils.get_itk_array(fn))
        img = feats_utils.get_itk_image(fn)
        prefix = os.path.basename(fn).split('.')[0]
        cen = feats.extract_centerlines(segmentation=data)
        rad = feats.extract_radius(segmentation=data, centerlines=cen)
        ofn = os.path.join(path_tmp_ves_sap, prefix + rad_suffix + fmt)
        feats.save_data(data=rad, img=img, filename=ofn)

    ###################################################################
    # METRIC GRAPH RECONSTRUCTION
    ###################################################################
    path = pathlib.Path(f"{path_tmp_ves_sap}/*_rads.nii.gz")
    filenames = list(sorted(pathlib.Path(path.parent).expanduser().glob(path.name)))
    if len(filenames) < n_jobs:
        n_jobs = len(filenames)
    print('\n----------------------------------------')
    print(f'Reconstructing metric graph with {n_jobs} job(s) ...')
    print('----------------------------------------')
    print('This may take a while ;)')

    # split filenames by #workers to parallelize metric graph extraction
    def split(a, n):
        k, m = divmod(len(a), n)
        return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))
    filenames = list(split(filenames, n_jobs))
    processes = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=metric_graph.run_reconstruction, args=(filenames[i],
                                                                                  path_visualization_metric_graph,
                                                                                  path_tmp_metric_graph))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    ###################################################################
    # EXTRACT GRAPH FEATURES
    ###################################################################
    print('\n----------------------------------------')
    print('Extracting graph features ...')
    print('----------------------------------------')
    # extract features from metric graph
    metric_graph_features_df = metric_graph.get_features(path_tmp_metric_graph, h_params=h_params["metric_graph"])

    ###################################################################
    # EXTRACT PYRADIOMICS FEATRUES
    ###################################################################
    print('\n----------------------------------------')
    print('Extracting pyradiomics features ...')
    print('----------------------------------------')
    # extract pyradiomics and construct Pandas dataframe
    image_filenames = pathlib.Path(path_tmp_vesselseg_prep_cut).glob("*")
    features = dict()
    for i_fn in tqdm(image_filenames, total=num_samples):
        mask_filenames = pathlib.Path(path_tmp_vesselseg_out_cut_clean).glob('*')
        m_fn = [str(m_fn) for m_fn in mask_filenames if "_".join(i_fn.name.split("_")[:3]) in m_fn.name][0]
        i_fn = str(i_fn)
        features[os.path.basename(i_fn)] = pyradiomics.get_pyradiomics_features(i_fn, m_fn)
    pyradiomics_df = pd.DataFrame.from_dict(features)

    # label samples as patient or volunteer
    is_patient = []
    for column in pyradiomics_df.columns:
        if "PAT" in column or "Pat" in column:
            is_patient.append(1)
        else:
            is_patient.append(0)
    pyradiomics_df = pyradiomics_df.T
    pyradiomics_df["is_patient"] = is_patient
    indexes = []
    for idx in pyradiomics_df.index:
        indexes.append("_".join(idx.split("_")[:3]))
    pyradiomics_df.index = indexes

    # save to tmp dir
    pyradiomics_df.to_csv(path_tmp_pyradiomics + "/pyradiomics.csv")

    ###################################################################
    # EXTRACT EPIDERMIS_WIDTH
    ###################################################################
    print('\n----------------------------------------')
    print('Extracting epidermis width ...')
    print('----------------------------------------')
    # extract minimal width of epidermis
    layerseg_filenames = pathlib.Path(path_tmp_layerseg_out).glob("*.nii.gz")
    columns = ["filename", "epidermis_width"]
    rows = []
    for fn in tqdm(layerseg_filenames, total=num_samples):
        layerseg = nib.load(str(fn)).get_fdata()
        epdermis_zs = []
        for i in range(layerseg.shape[2]):
            epdermis_zs.append(layerseg[:, :, i].sum() / layerseg.shape[1])
        rows.append(("_".join(fn.name.split("_")[:3]), min(epdermis_zs)))

    epidermis_width_df = pd.DataFrame(rows, columns=columns)
    epidermis_width_df = epidermis_width_df.set_index('filename')

    ###################################################################
    # CONSTRUCT FEATURE CSV
    ###################################################################
    # combine metric graph features, pyradiomics features and epidermis width
    df = pd.concat(
        [metric_graph_features_df, pyradiomics_df, epidermis_width_df], axis=1)

    # drop rows of samples, which have been detected as noise
    df = df.dropna()

    # encode label
    df = df.astype({'is_patient': 'int32'})

    # save dict as csv
    df.to_csv(f"{dirs['output']}/features.csv")

    print(f"Features were successfully saved to {dirs['output']}/features.csv")

    if delete_tmp:
        shutil.rmtree(os.path.join(dirs['output'], 'tmp'))


if __name__ == '__main__':
    dirs = {'input': './data/input',
            'laynet_model': './data/models/unet_depth5.pt',
            'vesnet_model': './data/models/vesnet_gn.pt',
            'output': './data/output'}

    h_params = {
        "segmentation": {
            "laynet_depth": 5,
            "ves_probability": 0.96973
        },
        "post_segmentation": {
            'min_size': 100,
            'epidermis_offset': 20,
            'roi_z': 150
        },
        "metric_graph": {
            'min_component_length': 100,
            'min_total_length': 300,
        }
    }

    dirs = {k: os.path.expanduser(v) for k, v in dirs.items()}

    vessel_pipeline(dirs=dirs,
                    h_params=h_params,
                    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    # pattern=['20200101000000'],  # if list, use these patterns
                    # otherwise whole directory
                    divs=(1, 1, 1),
                    delete_tmp=False,
                    n_jobs=10
                    )
