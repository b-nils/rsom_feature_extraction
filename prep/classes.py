from pathlib import Path

import os

import scipy.io as sio
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize_scalar

import imageio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import warnings

import nibabel as nib

from skimage import exposure, morphology, transform, filters


# CLASS FOR ONE DATASET
class Rsom():
    """
    class for preparing RSOM matlab data for layer and vessel segmentation
    """

    def __init__(self, filepathLF, filepathHF, filepathSURF='none'):
        """
        Create empty instance of RSOM
        """
        # if the filepaths are strings, generate PosixPath objects
        filepathLF = Path(filepathLF)
        filepathHF = Path(filepathHF)
        filepathSURF = Path(filepathSURF)

        # extact datetime number
        idx_1 = filepathLF.name.find('_')
        idx_2 = filepathLF.name.find('_', idx_1 + 1)
        DATETIME = filepathLF.name[idx_1:idx_2 + 1]

        # extract the 3 digit id + measurement string eg PAT001_RL01
        idxID = filepathLF.name.find('PAT')

        if idxID == -1:
            idxID = filepathLF.name.find('VOL')

        if idxID is not -1:
            ID = filepathLF.name[idxID:idxID + 11]
        else:
            # ID is different, extract string between Second "_" and third "_"
            # usually it is 6 characters long
            idx_3 = filepathLF.name.find('_', idx_2 + 1)
            ID = filepathLF.name[idx_2 + 1:idx_3]

        self.layer_end = None

        self.file = self.FileStruct(filepathLF, filepathHF, filepathSURF, ID, DATETIME)

    def prepare(self):

        self.read_matlab()
        self.flat_surface()
        self.cut_depth()
        self.norm_intensity()
        self.rescale_intensity()
        self.merge_volume_rgb()

    def read_matlab(self):
        '''
        read .mat files
        '''
        # load HF data
        self.matfileHF = sio.loadmat(self.file.HF)

        # extract high frequency Volume
        self.Vh = self.matfileHF['R']

        # load LF data
        self.matfileLF = sio.loadmat(self.file.LF)

        # extract low frequency Volume
        self.Vl = self.matfileLF['R']

        # load surface data
        try:
            self.matfileSURF = sio.loadmat(self.file.SURF)
        except:
            print(('WARNING: Could not load surface data, placing None type in'
                   'surface file. Method flatSURFACE is not going to be applied!!'))
            self.matfileSURF = None

    def flat_surface(self):
        '''
        modify volumetric data in order to get a flat skin surface
        options:
            override = True. If False, Volumetric data of the unflattened
            Skin will be saved.
        '''
        if self.matfileSURF is not None:

            # parse surface data and dx and dy
            S = self.matfileSURF['surfSmooth']
            dx = self.matfileSURF['dx']
            dy = self.matfileSURF['dy']

            # create meshgrid for surface data
            xSurf = np.arange(0, np.size(S, 0)) * dx
            ySurf = np.arange(0, np.size(S, 1)) * dy
            xSurf -= np.mean(xSurf)
            ySurf -= np.mean(ySurf)
            xxSurf, yySurf = np.meshgrid(xSurf, ySurf)

            # create meshgrid for volume data
            # use grid step dv
            # TODO: extract from reconParams
            # TODO: solve problem: python crashes when accessing reconParams
            dv = 0.012
            xVol = np.arange(0, np.size(self.Vl, 2)) * dv
            yVol = np.arange(0, np.size(self.Vl, 1)) * dv
            xVol -= np.mean(xVol)
            yVol -= np.mean(yVol)
            xxVol, yyVol = np.meshgrid(xVol, yVol)

            # generate interpolation function
            fn = interpolate.RectBivariateSpline(xSurf, ySurf, S)
            Sip = fn(xVol, yVol)

            Sip -= np.mean(Sip)

            # flip, to fit the grid
            Sip = Sip.transpose()

            self.Sip = Sip

            # for every surface element, calculate the offset
            # and shift volume elements perpendicular to surface
            for i in np.arange(np.size(self.Vl, 1)):
                for j in np.arange(np.size(self.Vl, 2)):

                    offs = int(-np.around(Sip[i, j] / 2))

                    self.Vl[:, i, j] = np.roll(self.Vl[:, i, j], offs);
                    self.Vh[:, i, j] = np.roll(self.Vh[:, i, j], offs);

                    # replace values rolled inside epidermis with zero
                    if offs < 0:
                        self.Vl[offs:, i, j] = 0
                        self.Vh[offs:, i, j] = 0

    def norm_intensity(self):
        '''
        normalize intensities to [0 1]
        '''

        self.Vl_1 = np.true_divide(self.Vl, np.amax(self.Vl))
        self.Vh_1 = np.true_divide(self.Vh, np.amax(self.Vh))

        self.Vl_1[self.Vl_1 > 1] = 1
        self.Vh_1[self.Vh_1 > 1] = 1
        self.Vl_1[self.Vl_1 < 0] = 0
        self.Vh_1[self.Vh_1 < 0] = 0

    def rescale_intensity(self, dynamic_rescale=False):
        '''
        rescale intensities, quadratic transform, crop values
        '''
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range=(0, 0.2))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range=(0, 0.1))

        self.Vl_1 = self.Vl_1 ** 2
        self.Vh_1 = self.Vh_1 ** 2

        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range=(0.05, 1))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range=(0.02, 1))

    def calc_mip(self, axis=1, do_plot=True, cut_z=0):
        '''
        plot maximum intensity projection along specified axis
        options:
            axis = 0,1,2. Axis along which to project
            do_plot = True. Plot into figure
            cut     = cut from volume in z direction
                      needed for on top view without epidermis

        '''
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0

        # maximum intensity projection
        self.Pl = np.amax(self.Vl[cut_z:, ...], axis=axis)
        self.Ph = np.amax(self.Vh[cut_z:, ...], axis=axis)

        # calculate alpha
        res = minimize_scalar(self.calc_alpha, bounds=(0, 100), method='bounded')
        alpha = res.x

        self.P = np.dstack([self.Pl, alpha * self.Ph, np.zeros(self.Ph.shape)])

        # cut negative values, in order to allow rescale to uint8
        self.P[self.P < 0] = 0

        self.P = exposure.rescale_intensity(self.P, out_range=np.uint8)
        self.P = self.P.astype(dtype=np.uint8)

        # rescale intensity
        val = np.quantile(self.P, (0.8, 0.9925))

        self.P = exposure.rescale_intensity(self.P,
                                            in_range=(val[0], val[1]),
                                            out_range=np.uint8)
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            plt.show()

    def calc_alpha(self, alpha):
        '''
        MIP helper function
        '''
        return np.sum(np.square(self.Pl - alpha * self.Ph))

    def merge_volume_rgb(self):
        '''
        merge low frequency and high frequency data feeding into different
        colour channels
        '''
        B = np.zeros(np.shape(self.Vl_1))

        self.Vm = np.stack([self.Vl_1, self.Vh_1, B], axis=-1)

    def cut_depth(self):
        '''
        cut Vl and Vh to 500 x 171 x 333
        '''
        zmax = 500

        # extract shape
        shp = self.Vl.shape

        if shp[0] >= zmax:
            self.Vl = self.Vl[:500, :, :]
            self.Vh = self.Vh[:500, :, :]
        else:
            ext = zmax - shp[0]
            # print('Extending volume. old shape:', shp)

            self.Vl = np.concatenate([self.Vl, np.zeros((ext, shp[1], shp[2]))], axis=0)
            self.Vh = np.concatenate([self.Vh, np.zeros((ext, shp[1], shp[2]))], axis=0)

            # print('New shape:', self.Vl.shape)

    def save_volume(self, destination, fstr=''):
        '''
        save rgb volume
        '''

        self.Vm = exposure.rescale_intensity(self.Vm, out_range=np.uint8)

        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.Vm.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        self.Vm = self.Vm.astype('u1')
        self.Vm = self.Vm.copy().view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.Vm, np.eye(4))

        # generate Path object
        destination = Path(destination)

        # if this is a cut file, need to construct the cut z-value
        # this is only for vesnet preparation
        if self.layer_end is not None:
            z_cut = '_' + 'z' + str(self.layer_end)
        else:
            z_cut = ''

        # generate filename
        nii_file = (destination / ('R' +
                                   self.file.DATETIME +
                                   self.file.ID +
                                   z_cut +
                                   '_' +
                                   fstr +
                                   '.nii.gz')).resolve()
        # print(str(nii_file))

        nib.save(img, str(nii_file))

    class FileStruct():
        def __init__(self, filepathLF, filepathHF, filepathSURF, ID, DATETIME):
            self.LF = filepathLF
            self.HF = filepathHF
            self.SURF = filepathSURF
            self.ID = ID
            self.DATETIME = DATETIME

    @staticmethod
    def loadNII(path):
        img = nib.load(path)
        return img.get_fdata()


class RsomVessel(Rsom):
    '''
    additional methods for preparing RSOM data for vessel segmentation,
    e.g. cut away epidermis
    '''

    def prepare(self, path, mode='pred', fstr='pred.nii.gz'):
        self.read_matlab()
        self.flat_surface()
        self.cut_depth()
        self.cut_layer(path, mode=mode, fstr=fstr)
        self.norm_intensity()
        self.rescale_intensity()
        self.merge_volume_rgb()

    def cut_layer(self, path, mode='pred', fstr='layer_pred.nii.gz'):
        '''
        cut off the epidermis with loading corresponding segmentation mask.
        '''

        # generate path
        filename = 'R' + self.file.DATETIME + self.file.ID + '_' + fstr
        file = os.path.join(path, filename)

        # print('Loading', file)

        # two modes supported, extract from prediction volume
        # or manual input through file
        if mode == 'pred':

            img = nib.load(file)
            self.S = img.get_fdata()
            self.S = self.S.astype(np.uint8)

            assert self.Vl.shape == self.S.shape, 'Shapes of raw and segmentation do not match'

            # print(self.Vl.shape)
            # print(self.S.shape)

            # for every slice in x-y plane, calculate label sum
            label_sum = np.sum(self.S, axis=(1, 2))

            max_occupation = np.amax(label_sum) / (self.S.shape[1] * self.S.shape[2])
            max_occupation_idx = np.argmax(label_sum)

            # print('Max occ', max_occupation)
            # print('idx max occ', max_occupation_idx)
            if max_occupation >= 0.01:
                # normalize
                label_sum = label_sum.astype(np.double) / np.amax(label_sum)

                # define cutoff parameter
                cutoff = 0.05

                label_sum_bin = label_sum > cutoff

                label_sum_idx = np.squeeze(np.nonzero(label_sum_bin))

                layer_end = label_sum_idx[-1]

                # additional fixed pixel offset
                # due to noise (reflection or poorly cut epidermis + 20)
                offs = 30  # 10
                layer_end += offs
            else:
                print("WARNING:  Could not determine valid epidermis layer.")
                layer_end = 0

        elif mode == 'manual':
            f = open(file)
            layer_end = int(str(f.read()))
            f.close()
        else:
            raise NotImplementedError

        layer_begin = layer_end + 150

        # print('Removing epidermis. Cutting at', layer_end)

        # cut away epidermis and lower part, so that every volume has same dimensions
        self.Vl = self.Vl[layer_end:layer_begin, :, :]
        self.Vh = self.Vh[layer_end:layer_begin, :, :]

        self.layer_end = layer_end

    def rescale_intensity(self):
        '''
        overrides method in class RSOM, because vessel segmentation needs
        different rescale
        '''
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range=(0, 0.25))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range=(0, 0.15))
        # self.Vl_1[:,:,:] = 0

        self.Vl_1 = self.Vl_1 ** 2
        self.Vh_1 = self.Vh_1 ** 2

        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range=(0.05, 1))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range=(0.05, 1))


class RsomVisualization(Rsom):
    '''
    subclass of RSOM
    for various visualization tasks
    '''

    def load_seg(self, filename):
        '''
        load segmentation file (can be labeled or predicted)

        '''
        return (self.loadNII(filename)).astype(np.uint8)

    def calc_mip_ves_seg(self, seg, axis=1, padding=(0, 0)):
        seg = self.load_seg(seg)

        mip = np.sum(seg, axis=axis) >= 1

        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool),
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)

        # naming convention: self.P is "normal" MIP
        #                    self.P_seg is segmentation MIP
        self.axis_P = axis
        self.P_seg = mip

    def calc_mip_lay_seg(self, seg, axis=1, padding=(0, 0)):
        seg = self.load_seg(seg)

        mip = np.sum(seg, axis=axis) >= 1

        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool),
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)

        # naming convention: self.P is "normal" MIP
        #                    self.P_seg is segmentation MIP
        self.axis_P = axis
        self.P_seg = mip

    def merge_mip_ves(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''

        P_seg = self.P_seg.astype(np.float32)
        P_seg_edge = filters.sobel(P_seg)
        P_seg_edge = P_seg_edge / np.amax(P_seg_edge)

        # feed into blue channel
        # enhanced edges
        blue = 150 * P_seg + 30 * P_seg_edge

        self.P_overlay = self.P.copy().astype(np.float32)
        self.P_overlay[:, :, 2] += blue
        self.P_overlay[self.P > 255] = 255

        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            plt.show()

    def merge_mip_lay(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''
        # use a semitransparent overlay
        self.P_overlay = _overlay(self.P_overlay, self.P_seg.astype(np.float32),
                                  alpha=0.5)

        self.P_overlay[self.P > 255] = 255

        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            plt.show()

    def _rescale_mip(self, scale):

        self.P_overlay = self.P_overlay.astype(np.uint8)

        if scale != 1:
            self.P = transform.rescale(self.P, scale, order=3, multichannel=True)

            # strangely transform.rescale is not dtype consistent?
            self.P = exposure.rescale_intensity(self.P, out_range=np.uint8)
            self.P = self.P.astype(np.uint8)

            self.P_overlay = transform.rescale(self.P_overlay, scale, order=3, multichannel=True)
            self.P_overlay = exposure.rescale_intensity(self.P_overlay, out_range=np.uint8)
            self.P_overlay = self.P_overlay.astype(np.uint8)

    def return_mip(self, scale=2):

        self._rescale_mip(scale)

        return self.P, self.P_overlay

    def save_comb_mip(self, dest, scale=2):

        self._rescale_mip(scale)

        if self.P.shape[0] > self.P.shape[1]:
            axis = 1
        else:
            axis = 0

        grey = 50

        img = np.concatenate((np.pad(self.P,
                                     ((2, 2), (2, 2), (0, 0)),
                                     mode='constant',
                                     constant_values=grey),
                              np.pad(self.P_overlay,
                                     ((2, 2), (2, 2), (0, 0)),
                                     mode='constant',
                                     constant_values=grey)),
                             axis=axis)
        img = np.pad(img,
                     ((2, 2), (2, 2), (0, 0)),
                     mode='constant',
                     constant_values=grey)

        img_file = os.path.join(dest, 'R' +
                                self.file.DATETIME +
                                self.file.ID +
                                '_' +
                                'combMIP_ax' +
                                str(self.axis_P) +
                                '.png')

        imageio.imwrite(img_file, img)


def _overlay(data, seg, alpha=0.5, colour=[255, 255, 255]):
    seg_mask = seg.copy()
    seg_mask = np.dstack((seg_mask, seg_mask, seg_mask)).astype(np.bool)
    seg_rgb = np.dstack((colour[0] * seg, colour[1] * seg, colour[2] * seg))

    ol = (alpha * seg_mask.astype(np.uint8) * data) + \
         (np.logical_not(seg_mask).astype(np.uint8) * data) + \
         (1 - alpha) * seg_rgb

    ol[ol > 255] = 255
    ol[ol < 0] = 0
    ol = ol.astype(np.uint8)
    return ol
