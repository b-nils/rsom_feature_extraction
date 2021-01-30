import os
import copy
import numpy as np
from skimage.morphology import skeletonize_3d
import nibabel as nib
import warnings

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from .patch_handling import get_patch

class RSOMVesselDataset(Dataset):
    """
    rsom dataset class for vessel segmentation
    
    Args:
        root_dir (string): Directory with all the nii.gz files.
        data_str (string): end part of filename of training data.
        label_str (string): end part of filename of segmentation ground truth data.
        transform (callable, optional): Optional transform to be applied
                            on a sample.
    """

    def __init__(self, 
                 root_dir, 
                 data_str='_v_rgb.nii.gz', 
                 label_str='_v_l.nii.gz',
                 divs = (1, 1, 1),
                 offset = (0, 0, 0),
                 transform=None):


        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str) and isinstance(label_str, str), \
        'data_str or label_str not valid.'
        
        self.data_str = data_str
        self.label_str = label_str
    
        self.divs = divs
        self.offset = offset
        
        # settings
        # obsolete, if multiprocessing in dataloader is used
        self.keep_last_data = False
        self.last_data_idx = -1
        
        
        # get all files in root_dir
        all_files = os.listdir(path = root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]
        label_len = len([el for el in all_files if el[-len(label_str):] == label_str])
        # if label_len is zero, set unlabeled flag
        if label_len is not 0:
            assert len(self.data) == \
                len([el for el in all_files if el[-len(label_str):] == label_str]), \
                'Amount of data and label files not equal.'
            self.unlabeled_flag = False
        else:
            self.unlabeled_flag = True
            #warnings.warn("Could not find any label files! Unlabeled dataset?", UserWarning)

    def __len__(self):
        return len(self.data) * np.prod(self.divs)
    
    def n_files(self):
        return len(self.data)
    
    @staticmethod
    def _readNII(rpath):
        '''
        read in the .nii.gz file
        Args:
            rpath (string)
        '''
        
        img = nib.load(str(rpath))
        
        # TODO: when does nib get_fdata() support rgb?
        # currently not, need to use old method get_data()
        return img.get_data()

    def __getitem__(self, idx):
        
        # remaining index
        rem_idx = np.mod(idx, np.prod(self.divs))
        
        # index of the volume
        data_idx = np.floor_divide(idx, np.prod(self.divs))
        
        assert data_idx*np.prod(self.divs) + rem_idx == idx
        
        # add meta information
        meta = {'filename': self.data[data_idx],
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None},
                'index': rem_idx}
        
        # load data, if neccessary
        if self.keep_last_data and data_idx == self.last_data_idx:
            data = self.last_data
            label = self.last_label
            
        else:
            data_path = os.path.join(self.root_dir, 
                                self.data[data_idx])
            label_path = os.path.join(self.root_dir, 
                                       self.data[data_idx].replace(self.data_str, self.label_str))
            # read data
            data = self._readNII(data_path)
            
            if isinstance(data.item(0), tuple):
                data = np.stack([data['R'], data['G'], data['B']], axis=-1)
            data = data.astype(np.float32)
            if not self.unlabeled_flag:
                # read label
                label = self._readNII(label_path)
                label = label.astype(np.float32)
            else:
                label = np.empty(data.shape[:-1], dtype=np.float32)
            
        if self.keep_last_data:
            if not data_idx == self.last_data_idx:
                # save the filename
                self.last_data_idx = data_idx
                
                # and data and label
                self.last_data = data
                self.last_label = label
            
        # crop data and label in order to be dividable by divs
        initial_dshape = data.shape
        initial_lshape = label.shape
        
        rem = np.mod(data.shape[:len(self.divs)], self.divs)
        
        assert len(rem) == 3, \
        'Other cases are not implemented. In general our data is 3D.'
        
        if rem[0]:
            data = data[int(np.floor(rem[0]/2)):-int(np.ceil(rem[0]/2)), ...]
            label = label[int(np.floor(rem[0]/2)):-int(np.ceil(rem[0]/2)), ...]
        
        if rem[1]:
            data = data[:, int(np.floor(rem[1]/2)):-int(np.ceil(rem[1]/2)), ...]
            label = label[:, int(np.floor(rem[1]/2)):-int(np.ceil(rem[1]/2)), ...]
            
        if rem[2]:
            data = data[:, :, int(np.floor(rem[2]/2)):-int(np.ceil(rem[2]/2)), ...]
            label = label[:, :, int(np.floor(rem[2]/2)):-int(np.ceil(rem[2]/2)), ...]
            
        # add to meta information, how much has been cropped
        meta['dcrop']['begin'] = torch.from_numpy(np.array(\
            [np.floor(rem[0]/2), np.floor(rem[1]/2), np.floor(rem[2]/2)], dtype=np.int16))
        meta['dcrop']['end'] = torch.from_numpy(np.array(\
            [np.ceil(rem[0]/2), np.ceil(rem[1]/2), np.ceil(rem[2]/2)], dtype=np.int16))
        
        # just for correct dimensionality
        # in case of RGB add another dimension
        if len(data.shape) == 4:
            meta['dcrop']['begin'] = torch.cat((meta['dcrop']['begin'], 
                torch.tensor([0], dtype=torch.int16)))
            meta['dcrop']['end'] = torch.cat((meta['dcrop']['end'], 
                torch.tensor([0], dtype=torch.int16)))
            
        meta['lcrop']['begin'] = torch.from_numpy(np.array(\
            [np.floor(rem[0]/2), np.floor(rem[1]/2), np.floor(rem[2]/2)], dtype=np.int16))
        meta['lcrop']['end'] = torch.from_numpy(np.array(\
            [np.ceil(rem[0]/2), np.ceil(rem[1]/2), np.ceil(rem[2]/2)], dtype=np.int16))

        assert np.all(np.array(initial_dshape) == meta['dcrop']['begin'].numpy()
                + meta['dcrop']['end'].numpy()
                + np.array(data.shape)),\
                'Shapes and Crop do not match'

        assert np.all(np.array(initial_lshape) == meta['lcrop']['begin'].numpy()
                + meta['lcrop']['end'].numpy()
                + np.array(label.shape)),\
                'Shapes and Crop do not match'
        
        #TODO in case the data is only 1D, fake dimension should also be added
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis=-1)
       
        # label always need expansion
        label = np.expand_dims(label, axis=-1)
    
        patch_data = get_patch(data, rem_idx, self.divs, self.offset)
        # don't need offset for label
        # after data passes CNN, if offset is chosen correctly, it matches
        # the shape of label exactly.
        patch_label = get_patch(label, rem_idx, self.divs, (0, 0, 0))

        sample = {'data': patch_data, 'label': patch_label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
class DropBlue():
    '''
    Drop the last slice of the RGB dimension
    RSOM images are 2channel, so blue is empty anyways.
    '''
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        if data.shape[-1] == 3:
            data = data[...,:2]
        else:
            warnings.warn('Calling DropBlue, even tho data is not RGB', UserWarning)

        assert data.shape[-1] == 2

        return {'data': data, 'label': label, 'meta': meta}
    
class ToTensor():
    '''
    Convert ndarrays in sample to tensors.
    '''
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        
        # data is [X1 x X2 x X3 x 2]
        # or      [X1 x X2 x X3]
        # label is [X1 x X2 x X3]
        
        # numpy array size of images
        # [X1... XN x C]
        # torch tensor size of images
        # [C x X1 ... XN]
        assert len(data.shape) == 4
        if len(data.shape) == 4:
            data = np.moveaxis(data, -1, 0)
            label = np.moveaxis(label, -1, 0)

        if 'label_skeleton' in meta:
            meta['label_skeleton'] = torch.from_numpy(np.moveaxis(meta['label_skeleton'], -1, 0))
        
        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label),
                'meta': meta}

def to_numpy(V, meta, Vtype='label', dimorder ='numpy'):
    '''
    inverse function for class ToTensor()
    args
        V: torch.tensor volume
        meta: batch['meta'] information

    return V as numpy.array volume
    '''
    
    if dimorder=='numpy':
        if isinstance(V, torch.Tensor):
            V = V.numpy()
    elif dimorder=='torch':
        if isinstance(V, torch.Tensor):
            V = V.numpy()
        # only in case RGB
        if len(V.shape) == 4:
            V = np.moveaxis(V, 0, -1)
    else:
        raise ValueError('Invalid argument for parameter dimorder')

    # add padding, which was removed before,
    # and saved in meta['lcrop'] and meta['dcrop']

    # structure for np.pad
    # (before0, after0), (before1, after1), ..)
    if Vtype=='label':
        # parse label crop
        b = (meta['lcrop']['begin']).numpy().squeeze()
        e = (meta['lcrop']['end']).numpy().squeeze()
    elif Vtype=='data':
        b = (meta['dcrop']['begin']).numpy().squeeze()
        e = (meta['dcrop']['end']).numpy().squeeze()
    else:
        raise ValueError('Invalid argument for parameter Vtype')
        
    # TODO: raise error if b, e is not dimension 3
    pad_width = ((b[0], e[0]), (b[1], e[1]), (b[2], e[2]))
    
    V = np.pad(V, pad_width, 'edge')

    return V
        
