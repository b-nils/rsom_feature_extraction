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

    # voxel volume in micrometer³
    features["total_blood_volume"] = shape.getVoxelVolumeFeatureValue() * 12 * 12 * 3 / 1000000000

    # avoid division by zero
    if features["total_blood_volume"] == 0:
        return features

    features["surface_volume_ratio"] = shape.getSurfaceVolumeRatioFeatureValue()

    return features
