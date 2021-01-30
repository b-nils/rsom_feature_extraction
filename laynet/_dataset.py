import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import copy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import nibabel as nib

class RSOMLayerDataset(Dataset):
    """
    rsom dataset class for layer segmentation
    
    Args:
        root_dir (string): Directory with all the nii.gz files.
        data_str (string): end part of filename of training data.
        label_str (string): end part of filename of segmentation ground truth data.
        transform (callable, optional): Optional transform to be applied
                            on a sample.
    """

    def __init__(self, 
                 root_dir, 
                 data_str='_rgb.nii.gz', 
                 label_str='_l.nii.gz',
                 slice_wise=False,
                 transform=None):

        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str) and isinstance(label_str, str), \
        'data_str or label_str not valid.'
        
        self.data_str = data_str
        self.label_str = label_str
        self.slice_wise = slice_wise
        
        # get all files in root_dir
        all_files = os.listdir(path = root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]

        if self.slice_wise:
            data_path = os.path.join(self.root_dir, 
                                     self.data[0])
            label_path = os.path.join(self.root_dir, 
                                      self.data[0].replace(self.data_str, self.label_str))
        
            # read data
            data = self._readNII(data_path)
            data = np.stack([data['R'], data['G'], data['B']], axis=-1)
            self.data_array = data.astype(np.float32)
        
            # read label
            label = self._readNII(label_path)
            self.label_array = label.astype(np.float32)

        
        assert len(self.data) == \
            len([el for el in all_files if el[-len(label_str):] == label_str]), \
            'Amount of data and label files not equal.'

    def __len__(self):
        if not self.slice_wise:
            return len(self.data)
        else:
            return self.data_array.shape[1]
    
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
        if not self.slice_wise:
            return self.getvolume(idx)
        else:
            return self.getslice(idx)
    
    def getvolume(self, idx):
        data_path = os.path.join(self.root_dir, 
                                 self.data[idx])
        label_path = os.path.join(self.root_dir, 
                                  self.data[idx].replace(self.data_str, self.label_str))
        
        # read data
        data = self._readNII(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        data = data.astype(np.float32)
        
        # read label
        label = self._readNII(label_path)
        label = label.astype(np.float32)
        
        # add meta information
        meta = {'filename': self.data[idx],
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None},
                'weight': 0}

        sample = {'data': data, 'label': label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getslice(self,idx):
        data = self.data_array[:,idx,...]
        data = np.expand_dims(data,1)
        
        label = self.label_array[:,idx,...]
        label = np.expand_dims(label,1)
        # add meta information
        meta = {'filename': self.data[0] + " slice_wise " + str(idx),
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None},
                'weight': 0}

        sample = {'data': data, 'label': label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample



class RSOMLayerDatasetUnlabeled(RSOMLayerDataset):
    """
    rsom dataset class for layer segmentation
    for prediction of unlabeled data only
    
    Args:
        root_dir (string): Directory with all the nii.gz files.
        data_str (string): end part of filename of training data
        transform (callable, optional): Optional transform to be applied
                            on a sample.
    """
    def __init__(self, root_dir, data_str='_rgb.nii.gz', transform=None):
        
        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
        'root_dir not a valid directory'
        
        self.root_dir = root_dir
        self.transform = transform
        
        assert isinstance(data_str, str), 'data_str or label_str not valid.'
        
        self.data_str = data_str
        # self.label_str = ''
        self.slice_wise = False        
        # get all files in root_dir
        all_files = os.listdir(path = root_dir)
        # extract the  data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]
        
        # assert len(self.data) == \
            # len([el for el in all_files if el[-len(label_str):] == label_str]), \
            # 'Amount of data and label files not equal.'

    def __getitem__(self, idx):
        data_path = os.path.join(self.root_dir, 
                            self.data[idx])
        # label_path = os.path.join(self.root_dir, 
        #                            self.data[idx].replace(self.data_str, self.label_str))
        
        # read data
        data = self._readNII(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        data = data.astype(np.float32)
        
        # read label
        # label = self._readNII(label_path)
        # label = label.astype(np.float32)
        label = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
        
        # add meta information
        meta = {'filename': self.data[idx],
                'dcrop':{'begin': None, 'end': None},
                'lcrop':{'begin': None, 'end': None},
                'weight': 0}

        sample = {'data': data, 'label': label, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, shuffle=False):
        self.shuffle = shuffle
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        
        ################ UPDATE
        # data can either RGB or RG

        # data is [Z x X x Y x 3] [500 x 171 x 333 x 3]
        # label is [Z x X x Y] [500 x 171 x 333]
        
        # we want one sample to be [Z x Y x 3]  2D rgb image
        
        # numpy array size of images
        # [H x W x C]
        # torch tensor size of images
        # [C x H x W]
        
        # and for batches
        # [B x C x H x W]
        
        # here, X is the batch size.
        # so we want to reshape to
        # [X x C x Z x Y] [171 x 3 x 500 x 333]
        data = data.transpose((1, 3, 0, 2))
        
        # and for the label
        # [X x Z x Y] [171 x 500 x 333]
        label = label.transpose((1, 0, 2))
        
        if data.shape[0] > 1 and self.shuffle:
           ds = data.shape
           ls = label.shape
           idx = torch.randperm(data.shape[0]).numpy()
           data = data[idx]
           label = label[idx]
           data = np.ascontiguousarray(data)
           label = np.ascontiguousarray(label)
           assert ds == data.shape
           assert ls == label.shape


        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return {'data': data.contiguous(),
                'label': label.contiguous(),
                'meta': meta}

class RandomZShift(object):
    """Apply random z-shift to sample.

    Args:
        max_shift (int, tuple of int):  maximum acceptable 
                                        shift in -z and +z direction (in voxel)
        
    """

    def __init__(self, max_shift=0):
        assert isinstance(max_shift, (int, tuple))
        if isinstance(max_shift, int):
            self.max_shift = (-max_shift, max_shift)
        else:
            assert len(max_shift) == 2
            assert max_shift[1] > max_shift[0]
            self.max_shift = max_shift

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        
        # initial shape
        data_ishape = data.shape
        label_ishape = label.shape
        
        # generate random dz offset
        dz = int(round((self.max_shift[1] - self.max_shift[0]) * torch.rand(1).item() + self.max_shift[0]))
        assert (dz >= self.max_shift[0] and dz <= self.max_shift[1])
         
        if dz:
            shift_data = np.zeros(((abs(dz), ) + data.shape[1:]), dtype = np.uint8)
            shift_label = np.zeros(((abs(dz), ) + label.shape[1:]), dtype = np.uint8)
        
            # print('RandomZShift: Check if this array modification does the correct thing before actually using it')
            # print('ZShift:', dz)
            # positive dz will shift in +z direction, "downwards" inside skin
            data = np.concatenate((shift_data, data[:-abs(dz),:,:,:])\
                    if dz > 0 else (data[abs(dz):,:,:,:], shift_data), axis = 0)
            label = np.concatenate((shift_label, label[:-abs(dz),:,:])\
                    if dz > 0 else (label[abs(dz):,:,:], shift_label), axis = 0)
            
            # data = np.concatenate((data[:-abs(dz),:,:,:], shift_data)\
            #         if dz > 0 else (shift_data, data[abs(dz):,:,:,:]), axis = 0)
            # label = np.concatenate((label[:-abs(dz),:,:], shift_label)\
            #         if dz > 0 else (shift_label, label[abs(dz):,:,:]), axis = 0)

            # should be the same...
            assert (data_ishape == data.shape and label_ishape == label.shape)
            data = np.ascontiguousarray(data)
            label = np.ascontiguousarray(label)
        return {'data': data, 'label': label, 'meta': meta}
    
class ZeroCenter(object):
    """ 
    Zero center input volumes
    """    
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3
        
        # compute for all x,y,z mean for every color channel
        # rgb_mean = np.around(np.mean(data, axis=(0, 1, 2))).astype(np.int16)
        # meanvec = np.tile(rgb_mean, (data.shape[:-1] + (1,)))
       
        # how to zero center??
        # data -= 127
        
        return {'data': data, 'label': label, 'meta': meta}
    
class DropBlue(object):
    """
    Drop the last slice of the RGB dimension
    RSOM images are 2channel, so blue is empty anyways.
    """
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3

        data = data[:,:,:,:2]

        assert data.shape[3] == 2

        return {'data': data, 'label': label, 'meta': meta}

class SwapDim(object):
    """
    swap x and y dimension to train network for the other view
    """
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3

        # data is [Z x X x Y x 3] [500 x 171 x 333 x 3]
        # label is [Z x X x Y] [500 x 171 x 333]
        
        data = np.swapaxes(data, 1, 2)
        label = np.swapaxes(label, 1, 2)
       
        return {'data': data, 'label': label, 'meta': meta}

class precalcLossWeight(object):
    """
    precalculation of a weight matrix used in the cross entropy
    loss function. It will be precalculated with the dataloader,
    so it can be computed in parallel
    call only after ToTensor!!
    """
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, torch.Tensor)

        # weight is meta['weight']

        #TODO: calculation
        target = label

        # LOSS shape [Minibatch, Z, X]
        target_shp = target.shape
        weight = copy.deepcopy(target)

 
        # loop over dim 0 and 2
        for yy in np.arange(target_shp[0]):
            for xx in np.arange(target_shp[2]):
                
                idx_nz = torch.nonzero(target[yy, :, xx])
                idx_beg = idx_nz[0].item()

                idx_end = idx_nz[-1].item()
                # weight[yy,:idx_beg,xx] = np.flip(scalingfn(idx_beg))
                # print(idx_beg, idx_end)
                
                A = self.scalingfn(idx_beg)
                B = self.scalingfn(target_shp[1] - idx_end)

                weight[yy,:idx_beg,xx] = A.unsqueeze(0).flip(1).squeeze()
                # print('A reversed', A.unsqueeze(0).flip(1).squeeze())
                # print('A', A)
                
                weight[yy,idx_end:,xx] = B
                # weight[yy,:idx_beg,xx] = np.flip(scalingfn(idx_beg))
                # weight[yy,idx_end:,xx] = scalingfn(label_shp[1] - idx_end)

        meta['weight'] = weight.float()

        return {'data': data, 'label': label, 'meta': meta}

    @staticmethod
    def scalingfn(l):
        '''
        l is length
        '''
        # linear, starting at 1
        y = torch.arange(l) + 1
        return y

class CropToEven(object):
    """ 
    if Volume shape is not even numbers, simply crop the first element
    except for last dimension, this is RGB  = 3
    """
    def __init__(self,network_depth=3):
        # how the unet works, without getting a upscaling error, the input shape must be a multiplier of 2**(network_depth-1)
        self.maxdiv = 2**(network_depth - 1)
        self.network_depth = network_depth

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
       
        # for backward compatibility
        # easy version: first crop to even, crop rest afterwards, if necessary
        initial_dshape = data.shape
        initial_lshape = label.shape

        IsOdd = np.mod(data.shape[:-1], 2)
        
        # hack, don't need to crop along what will be batch dimension later on
        IsOdd[1] = 0

        data = data[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:, : ]
        label = label[IsOdd[0]:, IsOdd[1]:, IsOdd[2]:]

        if not isinstance(meta['weight'], int):
            raise NotImplementedError('Weight was calulated before. Cropping implementation missing')
        
        # save, how much data was cropped
        # using torch tensor, because dataloader will convert anyways
        # dcrop = data
        meta['dcrop']['begin'] = torch.from_numpy(np.array([IsOdd[0], IsOdd[1], IsOdd[2], 0], dtype=np.int16))
        meta['dcrop']['end'] = torch.from_numpy(np.array([0, 0, 0, 0], dtype=np.int16))
        
        # lcrop = label
        meta['lcrop']['begin'] = torch.from_numpy(np.array([IsOdd[0], IsOdd[1], IsOdd[2]], dtype=np.int16))
        meta['lcrop']['end'] = torch.from_numpy(np.array([0, 0, 0], dtype=np.int16))

        
        # before cropping
        #            [Z  x Batch x Y  x 3]
        # data shape [500 x 171 x 333 x 3]
        # after cropping
        # data shape [500 x 170 x 332 x 3]

        # need to crop Z and Y
        
        # check if Z and Y are divisible through self.maxdiv
        rem0 = np.mod(data.shape[0], self.maxdiv)
        rem2 = np.mod(data.shape[2], self.maxdiv)
        
        if rem0 or rem2:
            if rem0:
                # crop Z
                data = data[int(np.floor(rem0/2)):-int(np.ceil(rem0/2)), :, :, :]
                label = label[int(np.floor(rem0/2)):-int(np.ceil(rem0/2)), :, :]

            if rem2:
                # crop Y
                data = data[ :, :, int(np.floor(rem2/2)):-int(np.ceil(rem2/2)), :]
                label = label[:, :, int(np.floor(rem2/2)):-int(np.ceil(rem2/2))]
        
            # add to meta information, how much has been cropped
            meta['dcrop']['begin'] += torch.from_numpy(np.array([np.floor(rem0/2), 0, np.floor(rem2/2), 0], dtype=np.int16))
            meta['dcrop']['end'] += torch.from_numpy(np.array([np.ceil(rem0/2), 0, np.ceil(rem2/2), 0], dtype=np.int16))
                
            meta['lcrop']['begin'] += torch.from_numpy(np.array([np.floor(rem0/2), 0, np.floor(rem2/2)], dtype=np.int16))
            meta['lcrop']['end'] += torch.from_numpy(np.array([np.ceil(rem0/2), 0, np.ceil(rem2/2)], dtype=np.int16))

        assert np.all(np.array(initial_dshape) == meta['dcrop']['begin'].numpy()
                + meta['dcrop']['end'].numpy()
                + np.array(data.shape)),\
                'Shapes and Crop do not match'

        assert np.all(np.array(initial_lshape) == meta['lcrop']['begin'].numpy()
                + meta['lcrop']['end'].numpy()
                + np.array(label.shape)),\
                'Shapes and Crop do not match'

        return {'data': data, 'label': label, 'meta': meta}
    
    
    

def to_numpy(V, meta):
    '''
    inverse function for class ToTensor() in dataloader_dev.py 
    args
        V: torch.tensor volume
        meta: batch['meta'] information

    return V as numpy.array volume
    '''
    # torch sizes X is batch size, C is Colour
    # data
    # [X x C x Z x Y] [171 x 3 x 500-crop x 333] (without crop)
    # and for the label
    # [X x Z x Y] [171 x 500 x 333]
    
    # we want to reshape to
    # numpy sizes
    # data
    # [Z x X x Y x 3] [500 x 171 x 333 x 3]
    # label
    # [Z x X x Y] [500 x 171 x 333]
    
    # here: we only need to backtransform labels
    if not isinstance(V, np.ndarray):
        assert isinstance(V, torch.Tensor)
        V = V.numpy()
    V = V.transpose((1, 0, 2))

    # add padding, which was removed before,
    # and saved in meta['lcrop'] and meta['dcrop']

    # structure for np.pad
    # (before0, after0), (before1, after1), ..)
    
    # parse label crop
    b = (meta['lcrop']['begin']).numpy().squeeze()
    e = (meta['lcrop']['end']).numpy().squeeze()
    # print('b, e')
    # print(b, e)
    # print(b.shape, e.shape)
    
    pad_width = ((b[0], e[0]), (b[1], e[1]), (b[2], e[2]))
    # print(V.shape)
    
    V = np.pad(V, pad_width, 'edge')

    # print(V.shape)
    return V
