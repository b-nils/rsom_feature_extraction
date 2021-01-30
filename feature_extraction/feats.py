"""
Extract centrelines, bifurcation points and radius from mask.

from:
    github: https://github.com/vessap/vessap/blob/master/feats.py
    author: @jocpae, @giesekow
"""

from __future__ import print_function
import argparse
import os
import numpy as np
from skimage.morphology import skeletonize_3d
from scipy import ndimage as ndi

from .feats_utils import get_itk_array, make_itk_image, write_itk_image, get_itk_image


def extract_centerlines(segmentation):
    skeleton = skeletonize_3d(segmentation)
    skeleton.astype(dtype='uint8', copy=False)
    return skeleton


def extract_bifurcations(centerlines):
    a = centerlines
    a.astype(dtype='uint8', copy=False)
    sh = np.shape(a)
    bifurcations = np.zeros(sh,dtype='uint8')
    endpoints = np.zeros(sh,dtype='uint8')

    for x in range(1,sh[0]-1):
        for y in range(1,sh[1]-1):
            for z in range(1,sh[2]-1):
                if a[x,y,z]== 1:
                    local = np.sum([a[ x-1,  y-1,  z-1],
                    a[ x-1,  y-1,  z],
                    a[ x-1,  y-1,  z+1],
                    a[ x-1,  y,  z-1],
                    a[ x-1,  y,  z],
                    a[ x-1,  y,  z+1],
                    a[ x-1,  y+1,  z-1],
                    a[ x-1,  y+1,  z],
                    a[ x-1,  y+1,  z+1],
                    a[ x,  y-1,  z-1],
                    a[ x,  y-1,  z],
                    a[ x,  y-1,  z+1],
                    a[ x,  y,  z-1],
                    a[ x,  y,  z],
                    a[ x,  y,  z+1],
                    a[ x,  y+1,  z-1],
                    a[ x,  y+1,  z],
                    a[ x,  y+1,  z+1],
                    a[ x+1,  y-1,  z-1],
                    a[ x+1,  y-1,  z],
                    a[ x+1,  y-1,  z+1],
                    a[ x+1,  y,  z-1],
                    a[ x+1,  y,  z],
                    a[ x+1,  y,  z+1],
                    a[ x+1,  y+1,  z-1],
                    a[ x+1,  y+1,  z],
                    a[ x+1,  y+1,  z+1]])

                    if local > 3:
                        bifurcations[x,y,z] = 1

    bifurcations.astype(dtype='uint8', copy=False)
    endpoints.astype(dtype='uint8', copy=False)
    return bifurcations, endpoints


def extract_radius(segmentation, centerlines):
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(image,return_indices=False)
    radius_matrix = transf*skeleton
    return radius_matrix


def preprocess_data(data):
    data = data.astype(np.int)
    data = ndi.binary_closing(data, iterations=1).astype(np.int)
    data = np.asarray(ndi.binary_fill_holes(data), dtype='uint8')
    return data


def save_data(data, img, filename):
    out_img = make_itk_image(data, img)
    write_itk_image(out_img, filename)
