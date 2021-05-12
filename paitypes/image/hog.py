import cv2
import numpy as np
from skimage.feature import hog

from typing import List, Tuple

from paitypes.geometry.Shape import Shape

from . import Image
from .channel import image_channels


def _extract_hog_features(image: Image,
                          orientations: int,
                          pixels_per_cell: int,
                          cells_per_block: int) -> np.ndarray:
    return np.concatenate([hog(channel,
                               orientations=orientations,
                               pixels_per_cell=(pixels_per_cell,
                                                pixels_per_cell),
                               cells_per_block=(cells_per_block,
                                                cells_per_block),
                               block_norm='L2-Hys',
                               transform_sqrt=True,
                               feature_vector=True
                               ) for channel in image_channels(image)])


def _extract_spatial_features(image: Image) -> np.ndarray:
    shape = Shape.from_image(image)
    features: List[np.ndarray] = []

    while shape.height >= 2 and shape.width >= 2:
        shape = Shape(shape.height / 2, shape.width / 2)
        shape_int = (int(shape.height), int(shape.width))
        features += [cv2.resize(image, shape_int).ravel()]

    return np.concatenate(features)


def _extract_histogram_features(image: Image,
                                n_bins: int
                                ) -> np.ndarray:
    return np.concatenate([
        np.histogram(channel, bins=n_bins)[0]
        for channel in image_channels(image)])


def extract_hog_and_supporting_features(image: Image) -> np.ndarray:
    IMAGE_SIZE = (64, 64)
    ORIENTATIONS = 9
    PIXELS_PER_CELL = 8
    CELLS_PER_BLOCK = 2
    HISTOGRAM_BINS = 32

    resized_image = cv2.resize(image, IMAGE_SIZE)

    hog_features = _extract_hog_features(
        resized_image,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK)

    spatial_features = _extract_spatial_features(resized_image)

    histogram_features = _extract_histogram_features(
        resized_image, HISTOGRAM_BINS)

    features = [hog_features, spatial_features, histogram_features]

    return np.concatenate(features)
