"""
Extract volumetric features using https://pyradiomics.readthedocs.io/en/latest/.
"""

import SimpleITK as itk
import radiomics.shape as rs


def get_pyradiomics_features(i_fn, m_fn):
    """
    Extracts volumetric features from mask.
    """
    features = dict()

    reader = itk.ImageFileReader()
    reader.SetFileName(i_fn)
    image = reader.Execute()
    reader.SetFileName(m_fn)
    mask = reader.Execute()
    shape = rs.RadiomicsShape(image, mask)

    features["voxel_volume"] = shape.getVoxelVolumeFeatureValue()

    if features["voxel_volume"] == 0:
        return features

    features["surface_volume_ratio"] = shape.getSurfaceVolumeRatioFeatureValue()
    features["sphericity"] = shape.getSphericityFeatureValue()
    features["maximum_3d_diameter"] = shape.getMaximum3DDiameterFeatureValue()
    features["maximum_2d_diameter_slice"] = shape.getMaximum2DDiameterSliceFeatureValue()
    features["maximum_2d_diameter_column"] = shape.getMaximum2DDiameterColumnFeatureValue()
    features["maximum_2d_diameter_row"] = shape.getMaximum2DDiameterRowFeatureValue()

    return features
