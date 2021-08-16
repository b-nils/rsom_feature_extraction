import imageio
import scipy.ndimage
import torch
import numpy as np
import matplotlib.pyplot as plt

import shutil
import os
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import nibabel as nib
import concurrent.futures


def timing(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(func.__name__, 'took', t2 - t1, 's')

    return wrapper


class RsomLayerDataset(Dataset):
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
                 batch_size=2,
                 sliding_window_size=5,
                 training=True,
                 deterministic=False,
                 transform=None):

        super().__init__()

        assert os.path.exists(root_dir) and os.path.isdir(root_dir), \
            'root_dir not a valid directory'

        self.root_dir = root_dir
        self.transform = transform
        self.training = training
        self.sliding_window_size = sliding_window_size

        if self.training:
            self.batch_size = batch_size

        # for debug
        self.write_png = False

        assert isinstance(data_str, str) and isinstance(label_str, str), \
            'data_str or label_str not valid.'

        self.data_str = data_str
        self.label_str = label_str

        # get all files in root_dir
        all_files = os.listdir(path=root_dir)
        # extract the data files
        self.data = [el for el in all_files if el[-len(data_str):] == data_str]

        if self.training:
            assert len(self.data) == \
                   len([el for el in all_files if el[-len(label_str):] == label_str]), \
                'Amount of data and label files not equal.'

        if self.training:
            self.npz_root_dir = os.path.join(root_dir, 'npz_files')
            self.npz_data_x = []
            self.npz_data_y = []
            if not os.path.exists(self.npz_root_dir):
                os.mkdir(self.npz_root_dir)

            self._preprocess_files()

            # generate lists for batch size
            self.npz_data_x = np.asarray(self.npz_data_x)
            self.npz_data_y = np.asarray(self.npz_data_y)

            if deterministic:
                torch.manual_seed(23)
                np.random.seed(15)

            self.npz_data_x = self.npz_data_x[torch.randperm(len(self.npz_data_x)).numpy()]
            self.npz_data_y = self.npz_data_y[torch.randperm(len(self.npz_data_y)).numpy()]

            indices_x = [*range(0, len(self.npz_data_x), self.batch_size)]
            indices_y = [*range(0, len(self.npz_data_y), self.batch_size)]

            self.npz_batches_x = np.array_split(self.npz_data_x, indices_x[1:])
            self.npz_batches_y = np.array_split(self.npz_data_y, indices_y[1:])
            map(lambda x: x.tolist(), self.npz_batches_x)
            map(lambda x: x.tolist(), self.npz_batches_y)

            self.npz_batches = self.npz_batches_x + self.npz_batches_y

    def cleanup(self):
        if self.training:
            shutil.rmtree(self.npz_root_dir)

    @timing
    def _preprocess_files(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        results = executor.map(self._preprocess_file, self.data)

        for _ in results:
            # wait for threads to finish.
            pass

    def _preprocess_file(self, file):
        data_path = os.path.join(self.root_dir,
                                 file)
        label_path = os.path.join(self.root_dir,
                                  file.replace(self.data_str, self.label_str))

        # read data
        data = self._read_nii(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        # data = data.astype(np.float32)

        # read label
        label = self._read_nii(label_path)
        # label = label.astype(np.float32)

        data = data.astype(np.uint8)
        label = label.astype(np.uint8)

        data_mip_x, data_mip_y = self.sliding_window_mip(data)

        data_list_x = np.split(data_mip_x, data_mip_x.shape[1], axis=1)
        data_list_y = np.split(data_mip_y, data_mip_y.shape[2], axis=2)

        label_list_x = np.split(label, label.shape[1], axis=1)
        label_list_y = np.split(label, label.shape[2], axis=2)

        for idx, element in enumerate(zip(data_list_x, label_list_x)):
            data, label = element

            npz_file = os.path.join(self.npz_root_dir,
                                    file.replace(self.data_str,
                                                 '_xframe' + str(idx) + '.npz'))
            self.npz_data_x.append(npz_file)

            with open(npz_file, 'wb') as f:
                np.savez(npz_file, data=data, label=label)

            if self.write_png:
                imageio.imwrite(npz_file.replace('.npz', '_rgb.png'), np.squeeze(data))
                imageio.imwrite(npz_file.replace('.npz', '_l.png'), np.squeeze(label * 255))

        for idx, element in enumerate(zip(data_list_y, label_list_y)):
            # print('y', idx, data.shape, label.shape)
            data, label = element
            npz_file = os.path.join(self.npz_root_dir,
                                    file.replace(self.data_str,
                                                 '_yframe' + str(idx) + '.npz'))
            self.npz_data_y.append(npz_file)

            with open(npz_file, 'wb') as f:
                np.savez(npz_file, data=data, label=label)
            if self.write_png:
                imageio.imwrite(npz_file.replace('.npz', '_rgb.png'), np.squeeze(data))
                imageio.imwrite(npz_file.replace('.npz', '_l.png'), np.squeeze(label * 255))

    def sliding_window_mip(self, data):
        data_mip_x = scipy.ndimage.maximum_filter1d(data, size=self.sliding_window_size, axis=1)
        data_mip_y = scipy.ndimage.maximum_filter1d(data, size=self.sliding_window_size, axis=2)

        return data_mip_x, data_mip_y

    @staticmethod
    def _read_nii(rpath):
        '''
        read in the .nii.gz file
        '''

        img = nib.load(str(rpath))

        # TODO: when does nib get_fdata() support rgb?
        # currently not, need to use old method get_data()
        return img.get_data()

    @staticmethod
    def save_nii(sample_, path):

        if not isinstance(sample_, tuple):
            sample_ = (sample_,)

        for idx, sample in enumerate(sample_):

            data = sample['data']
            data = data.numpy()
            data = np.moveaxis(data, 1, -1)

            # add blue channel
            data = np.concatenate((data, np.zeros((data.shape[0], data.shape[1], data.shape[2], 1), dtype=np.uint8)),
                                  axis=-1)

            data = np.ascontiguousarray(data)

            label = sample['label']
            label = label.numpy()
            label = np.squeeze(label)

            shape_3d = data.shape[0:3]
            rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            data = data.astype('u1')
            data = data.view(rgb_dtype).reshape(shape_3d)

            img_data = nib.Nifti1Image(data, np.eye(4))

            if isinstance(sample['meta']['filename'], list):
                filename_meta = sample['meta']['filename'][0].replace('.npz', '.nii.gz')
            else:
                filename_meta = sample['meta']['filename']

            filename = os.path.join(path, os.path.basename(filename_meta))

            nib.save(img_data, filename.replace('.nii.gz', '_' + str(idx) + '_rgb.nii.gz'))

            label = label.astype(np.uint8)
            img_label = nib.Nifti1Image(label, np.eye(4))
            nib.save(img_label, filename.replace('.nii.gz', '_' + str(idx) + '_l.nii.gz'))

    def __len__(self):
        if self.training:
            return len(self.npz_batches)
        else:
            return len(self.data)

    def _getitem_train(self, idx):
        data = []
        label = []
        filenames = []
        for npz_file in self.npz_batches[idx]:
            # print(npz_file)
            with open(npz_file, 'rb') as f:
                tmp = np.load(npz_file)

            data.append(np.squeeze(tmp['data']))
            label.append(np.squeeze(tmp['label']))
            filenames.append(npz_file)

        # batch size is first axis
        data_batch = np.stack(data, axis=0)
        label_batch = np.stack(label, axis=0)

        if '_xframe' in os.path.basename(npz_file):
            batch_axis = 'x'
        elif '_yframe' in os.path.basename(npz_file):
            batch_axis = 'y'

        # add meta information
        meta = {'filename': filenames,
                'dcrop': {'begin': None, 'end': None},
                'lcrop': {'begin': None, 'end': None},
                'weight': 0,
                'batch_axis': batch_axis}

        sample = {'data': data_batch, 'label': label_batch, 'meta': meta}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        if self.training:
            return self._getitem_train(idx)
        else:
            return self._getitem_test(idx)

    def _getitem_test(self, idx):
        # TODO process both sides? x and y. either return list [x, y] or stack on top
        return self._getvolume(idx)

    def _getvolume(self, idx):
        data_path = os.path.join(self.root_dir,
                                 self.data[idx])
        label_path = os.path.join(self.root_dir,
                                  self.data[idx].replace(self.data_str, self.label_str))

        # read data
        data = self._read_nii(data_path)
        data = np.stack([data['R'], data['G'], data['B']], axis=-1)
        # data = data.astype(np.float32)

        # read label
        label = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)#self._read_nii(label_path)
        # label = label.astype(np.float32)

        data = data.astype(np.uint8)
        label = label.astype(np.uint8)

        data_mip_x, data_mip_y = self.sliding_window_mip(data)

        # 500 to axis 1
        data_mip_x = np.swapaxes(data_mip_x, 0, 1)
        label_mip_x = np.swapaxes(label, 0, 1)

        # 500 to axis 1
        data_mip_y = np.swapaxes(data_mip_y, 0, 1)
        label_mip_y = np.swapaxes(label, 0, 1)

        # swap
        data_mip_y = np.swapaxes(data_mip_y, 0, 2)
        label_mip_y = np.swapaxes(label_mip_y, 0, 2)

        # add meta information
        meta_x = {'filename': self.data[idx],
                  'dcrop': {'begin': None, 'end': None},
                  'lcrop': {'begin': None, 'end': None},
                  'weight': 0,
                  'batch_axis': 'x'}

        sample_x = {'data': data_mip_x, 'label': label_mip_x, 'meta': meta_x}

        meta_y = {'filename': self.data[idx],
                  'dcrop': {'begin': None, 'end': None},
                  'lcrop': {'begin': None, 'end': None},
                  'weight': 0,
                  'batch_axis': 'y'}

        sample_y = {'data': data_mip_y, 'label': label_mip_y, 'meta': meta_y}

        if self.transform:
            sample_x = self.transform(sample_x)
            sample_y = self.transform(sample_y)

        return sample_x, sample_y


class RandomZRescale:
    def __init__(self, p=0.5, range=(0.6, 1.5)):
        self.p = p
        self.range = range

    def rescale_one(self, data, label, scale):
        # find epidermis z-coords
        z_proj = np.amax(label, axis=1)
        nz = np.nonzero(z_proj)
        epidermis_start = nz[0][0]
        epidermis_end = nz[0][-1]
        data_chunks = []
        label_chunks = []

        data_chunks.append(data[:epidermis_start, :])
        label_chunks.append(label[:epidermis_start, :])

        # data_chunks.append(skimage.transform.rescale(data[epidermis_start:epidermis_end, :],
        #                                              scale=(scale, 1, 1),
        #                                              order=1  # bi-linear
        #                                              ))

        data_chunks.append(scipy.ndimage.zoom(data[epidermis_start:epidermis_end, :],
                                              zoom=(scale, 1, 1),
                                              order=1))

        # label_chunks.append(skimage.transform.rescale(label[epidermis_start:epidermis_end, :],
        #                                               order=0,  # nearest neighbor
        #                                               scale=(scale, 1),
        #                                               preserve_range=True))

        label_chunks.append(scipy.ndimage.zoom(label[epidermis_start:epidermis_end, :],
                                               zoom=(scale, 1),
                                               order=0))

        # only append if we are still smaller than 500
        if sum([el.shape[0] for el in label_chunks]) < 500:
            data_chunks.append(data[epidermis_end:, ...])
            label_chunks.append(label[epidermis_end:, ...])

        sum_z = sum([el.shape[0] for el in label_chunks])

        if sum_z > 500:
            data_chunks[-1] = data_chunks[-1][:-(sum_z - 500), ...]
            label_chunks[-1] = label_chunks[-1][:-(sum_z - 500), ...]
        elif sum_z < 500:
            data_chunks.append(np.zeros((500 - sum_z, data.shape[1], data.shape[2]), dtype=data.dtype))
            label_chunks.append(np.zeros((500 - sum_z, label.shape[1]), dtype=label.dtype))

        data = np.concatenate(data_chunks, axis=0)
        label = np.concatenate(label_chunks, axis=0)
        try:
            assert data.shape[0] == 500
            assert label.shape[0] == 500
        except Exception as e:
            print(scale)
            print(e)
            print(data.shape)
            print(label.shape)
            for el in data_chunks:
                print(el.shape)
            for el in label_chunks:
                print(el.shape)

        return data, label

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3
        assert data.shape[1] == 500
        assert label.shape[1] == 500

        rand_vec = torch.rand(data.shape[0]).numpy()

        for idx in range(len(rand_vec)):
            if rand_vec[idx] <= self.p:
                # if True:
                # sample scale
                scale = torch.rand(1).numpy() * (self.range[1] - self.range[0]) + self.range[0]
                scale = scale[0]
                # print(f"{scale=}")
                data[idx, ...], label[idx, ...] = self.rescale_one(data[idx, ...], label[idx, ...], scale=scale)

        return {'data': np.ascontiguousarray(data.astype(np.uint8)), 'label': np.ascontiguousarray(label), 'meta': meta}


class RandomMirror:
    """
    Mirror slices.
    """

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3
        assert data.shape[1] == 500
        assert label.shape[1] == 500

        rand_vec = torch.rand(data.shape[0]).numpy()

        for idx in range(len(rand_vec)):
            if rand_vec[idx] > 0.5:
                data[idx, ...] = np.flip(data[idx, ...], axis=1)
                label[idx, ...] = np.flip(label[idx, ...], axis=1)

        return {'data': data, 'label': label, 'meta': meta}


class IntensityTransform:
    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        # data still is RGB
        assert data.shape[3] == 3
        assert data.dtype == np.uint8

        # intensity transform
        x_rand = torch.randn(1).item()
        y_rand = torch.randn(1).item()

        y_max = 255
        x_max = 255

        # variance
        x_rand /= 5
        y_rand /= 5

        # mean
        x_rand += 0.5
        y_rand += 0.5

        # clip
        clip = 0.2
        x_rand = x_rand if x_rand > clip else clip
        y_rand = y_rand if y_rand > clip else clip

        x_rand = x_rand if x_rand < 1 - clip else clip
        y_rand = y_rand if y_rand < 1 - clip else clip

        x_rand *= x_max
        y_rand *= y_max

        m1 = y_rand / x_rand
        b1 = y_rand - m1 * x_rand

        m2 = (y_max - y_rand) / (x_max - x_rand)
        b2 = y_max - m2 * x_max

        data = np.piecewise(data,
                            [data < x_rand, data >= x_rand],
                            [lambda x: m1 * x + b1, lambda x: m2 * x + b2]
                            )

        return {'data': data, 'label': label, 'meta': meta}


class Normalize:
    """
    normalize
    """

    def __init__(self, symmetric=False):
        self.symmetric = symmetric

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)
        assert data.dtype == np.uint8

        # data still is RGB
        assert data.shape[3] == 3

        data = data.astype(np.float32)
        if self.symmetric:
            data -= 127.5
            data /= 127.5
        else:
            data /= 255

        return {'data': data, 'label': label, 'meta': meta}


class RandomZShift:
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

        # for every slice in batch
        for idx in range(data.shape[0]):

            dz = int(round((self.max_shift[1] - self.max_shift[0]) * torch.rand(1).item() + self.max_shift[0]))
            assert dz >= self.max_shift[0] and dz <= self.max_shift[1]

            if dz:
                data_sample = data[idx, ...]
                label_sample = label[idx, ...]

                shift_data = np.zeros(((abs(dz),) + data_sample.shape[1:]), dtype=np.uint8)
                shift_label = np.zeros(((abs(dz),) + label_sample.shape[1:]), dtype=np.uint8)

                # print('ZShift:', dz)
                # positive dz will shift in +z direction, "downwards" inside skin
                data_sample = np.concatenate((shift_data, data_sample[:-abs(dz), ...]) \
                                                 if dz > 0 else (data_sample[abs(dz):, ...], shift_data), axis=0)
                label_sample = np.concatenate((shift_label, label_sample[:-abs(dz), ...]) \
                                                  if dz > 0 else (label_sample[abs(dz):, ...], shift_label), axis=0)

                # should be the same...
                assert (data_ishape == data.shape and label_ishape == label.shape)

                data[idx, ...] = data_sample
                label[idx, ...] = label_sample

        data = np.ascontiguousarray(data)
        label = np.ascontiguousarray(label)

        return {'data': data, 'label': label, 'meta': meta}


class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, shuffle=False):
        self.shuffle = shuffle

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']

        # torch channel ordering
        # [ BATCH x CHANNELS x X1 x X2]

        # shape is:
        # BATCH x 500 x {171,333} x 3

        # drop blue channel

        if data.shape[-1] == 3:
            data = data[..., :2]

        # move channel axis
        data = np.moveaxis(data, -1, 1)

        # create singleton channel axis
        label = np.expand_dims(label, 1)

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        return {'data': data.contiguous(),
                'label': label.contiguous(),
                'meta': meta}


class CropToEven:

    def __init__(self, network_depth=3):
        # how the unet works, without getting a upscaling error, the input shape must be a multiplier of 2**(network_depth-1)
        self.maxdiv = 2 ** (network_depth - 1)
        self.network_depth = network_depth

    def __call__(self, sample):
        data, label, meta = sample['data'], sample['label'], sample['meta']
        assert isinstance(data, np.ndarray)
        assert isinstance(label, np.ndarray)

        assert data.shape[-1] == 3

        # for backward compatibility
        # easy version: first crop to even, crop rest afterwards, if necessary
        initial_dshape = data.shape
        initial_lshape = label.shape

        IsOdd = np.mod(data.shape[:-1], 2)

        # batch dimension is first dimension
        data = data[:, IsOdd[1]:, IsOdd[2]:, :]
        label = label[:, IsOdd[1]:, IsOdd[2]:]

        # save, how much data was cropped
        # using torch tensor, because dataloader will convert anyways
        # dcrop = data
        meta['dcrop']['begin'] = torch.from_numpy(np.array([0, IsOdd[1], IsOdd[2], 0], dtype=np.int16))
        meta['dcrop']['end'] = torch.from_numpy(np.array([0, 0, 0, 0], dtype=np.int16))

        # lcrop = label
        meta['lcrop']['begin'] = torch.from_numpy(np.array([0, IsOdd[1], IsOdd[2]], dtype=np.int16))
        meta['lcrop']['end'] = torch.from_numpy(np.array([0, 0, 0], dtype=np.int16))

        # check if Z and Y are divisible through self.maxdiv
        rem1 = np.mod(data.shape[1], self.maxdiv)
        rem2 = np.mod(data.shape[2], self.maxdiv)

        if rem1 or rem2:
            if rem1:
                # crop Z
                data = data[:, int(np.floor(rem1 / 2)):-int(np.ceil(rem1 / 2)), :, :]
                label = label[:, int(np.floor(rem1 / 2)):-int(np.ceil(rem1 / 2)), :]

            if rem2:
                # crop Y
                data = data[:, :, int(np.floor(rem2 / 2)):-int(np.ceil(rem2 / 2)), :]
                label = label[:, :, int(np.floor(rem2 / 2)):-int(np.ceil(rem2 / 2))]

            # add to meta information, how much has been cropped
            meta['dcrop']['begin'] += torch.from_numpy(
                np.array([0, np.floor(rem1 / 2), np.floor(rem2 / 2), 0], dtype=np.int16))
            meta['dcrop']['end'] += torch.from_numpy(
                np.array([0, np.ceil(rem1 / 2), np.ceil(rem2 / 2), 0], dtype=np.int16))

            meta['lcrop']['begin'] += torch.from_numpy(
                np.array([0, np.floor(rem1 / 2), np.floor(rem2 / 2)], dtype=np.int16))
            meta['lcrop']['end'] += torch.from_numpy(
                np.array([0, np.ceil(rem1 / 2), np.ceil(rem2 / 2)], dtype=np.int16))

        assert np.all(np.array(initial_dshape) == meta['dcrop']['begin'].numpy()
                      + meta['dcrop']['end'].numpy()
                      + np.array(data.shape)), \
            'Shapes and Crop do not match'

        assert np.all(np.array(initial_lshape) == meta['lcrop']['begin'].numpy()
                      + meta['lcrop']['end'].numpy()
                      + np.array(label.shape)), \
            'Shapes and Crop do not match'

        return {'data': data, 'label': label, 'meta': meta}


def to_numpy(volume, meta):
    '''
    inverse function for class ToTensor() in dataloader_dev.py
    args
        V: torch.tensor volume
        meta: batch['meta'] information
    return V as numpy.array volume
    '''

    # we want to reshape to
    # numpy sizes
    # data
    # [Z x X x Y x 3] [500 x 171 x 333 x 3]
    # label
    # [Z x X x Y] [500 x 171 x 333]

    # here: we only need to backtransform labels
    if not isinstance(volume, np.ndarray):
        assert isinstance(volume, torch.Tensor)
        volume = volume.numpy()

    volume = np.squeeze(volume, axis=1)

    # add padding, which was removed before,
    # and saved in meta['lcrop'] and meta['dcrop']

    # structure for np.pad
    # (before0, after0), (before1, after1), ..)

    # parse label crop
    b = (meta['lcrop']['begin']).numpy().squeeze()
    e = (meta['lcrop']['end']).numpy().squeeze()

    pad_width = list(zip(b, e))

    volume = np.pad(volume, pad_width, 'edge')

    batch_axis = meta['batch_axis']
    if isinstance(batch_axis, list):
        batch_axis = batch_axis[0]

    if batch_axis == 'x':
        volume = volume.transpose((1, 0, 2))
    elif batch_axis == 'y':
        volume = volume.transpose((1, 2, 0))
    else:
        raise AttributeError

    return np.ascontiguousarray(volume)


def show(element, idx=0):
    import matplotlib.pyplot as plt
    data = element['data']

    plt.figure()
    plt.imshow(data[idx, ...])
    plt.figure()
    plt.imshow(element['label'][idx, ...], cmap='Greys')