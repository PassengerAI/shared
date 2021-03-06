# -*- coding: utf-8 -*-
# Taken from https://github.com/colincsl/pyKinectTools/blob/master/pyKinectTools/algs/HistogramOfOpticalFlow.py  # noqa

import numpy as np
from scipy import sqrt, arctan2, pi
from scipy.ndimage import uniform_filter
from typing import Tuple, no_type_check
from paitypes.image.optical_flow import OpticalFlowImage

from typing import NewType


class HOOFError(ValueError):
    pass


HOOF = NewType('HOOF', np.ndarray)


# XXX: Since this code comes from external, skip internal type checking
@no_type_check
def generate_hoof(flow: OpticalFlowImage,
                  orientations: int = 5,
                  pixels_per_cell: Tuple[int, int] = (8, 8),
                  cells_per_block: Tuple[int, int] = (3, 3),
                  normalise: bool = False,
                  motion_threshold: float = 1.0
                  ) -> HOOF:
    """Extract Histogram of Optical Flow (HOF) for a given image.
    Key difference between this and HOG is that flow is MxNx2 instead of MxN
    Compute a Histogram of Optical Flow (HOF) by
        1. (optional) global image normalisation
        2. computing the dense optical flow
        3. computing flow histograms
        4. normalising across blocks
        5. flattening into a feature vector
    Parameters
    ----------
    Flow : (M, N) ndarray
        Input image (x and y flow images).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    normalise : bool, optional
        Apply power law compression to normalise the image before
        processing.
    static_threshold : threshold for no motion
    Returns
    -------
    newarr : ndarray
        hof for the image as a 1D (flattened) array.
    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA
    """
    flow = np.atleast_2d(flow)

    """
    -1-
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    if flow.ndim < 3:
        raise HOOFError("Requires dense flow in both directions")

    if normalise:
        flow = sqrt(flow)

    """
    -2-
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if flow.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        flow = flow.astype('float')

    gx = np.zeros(flow.shape[:2])
    gy = np.zeros(flow.shape[:2])
    # gx[:, :-1] = np.diff(flow[:,:,1], n=1, axis=1)
    # gy[:-1, :] = np.diff(flow[:,:,0], n=1, axis=0)

    gx = flow[:, :, 1]
    gy = flow[:, :, 0]

    """
    -3-
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx**2 + gy**2)
    orientation = arctan2(gy, gx) * (180 / pi) % 180

    sy, sx = flow.shape[:2]
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))
    subsample = np.index_exp[cy / 2:cy * n_cellsy:cy, cx / 2:cx * n_cellsx:cx]

    # XXX/drs: Fix float precision problem in slices
    def fix_slice(slice_):
        return slice(int(slice_.start), int(slice_.stop), int(slice_.step))
    subsample_fixed = (fix_slice(subsample[0]), fix_slice(subsample[1]))
    subsample = subsample_fixed
    # XXX/drs: End patch

    for i in range(orientations - 1):
        # create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / orientations * (i + 1),
                            orientation, -1)
        temp_ori = np.where(orientation >= 180 / orientations * i,
                            temp_ori, -1)
        # select magnitudes for those orientations
        cond2 = (temp_ori > -1) * (magnitude > motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)

        temp_filt = uniform_filter(temp_mag, size=(cy, cx))
        orientation_histogram[:, :, i] = temp_filt[subsample]

    ''' Calculate the no-motion bin '''
    temp_mag = np.where(magnitude <= motion_threshold, magnitude, 0)

    temp_filt = uniform_filter(temp_mag, size=(cy, cx))
    orientation_histogram[:, :, -1] = temp_filt[subsample]

    # now for each cell, compute the histogram
    hof_image = None

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (hog) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5
            # XXX/drs: This code is too slow and we had to optimize it.
            # normalised_blocks[y, x, :] = block / sqrt(block.sum()**2 + eps)
            normalised_blocks[y, x, :] = block

    # XXX/drs: Begin optimizations
    sums = sqrt(normalised_blocks.sum(axis=(2, 3, 4))**2 + eps)
    normalised_blocks /= sums[:, :, None, None, None]
    # XXX/drs: End optimizations

    """
    The final step collects the hof descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    return HOOF(normalised_blocks.ravel())
