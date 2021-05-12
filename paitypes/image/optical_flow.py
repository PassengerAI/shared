import cv2
import numpy as np

from typing import NewType

from paitypes.image.color import GrayscaleImage


OpticalFlowImage = NewType('OpticalFlowImage', np.ndarray)


def calculate_optical_flow(frame1: GrayscaleImage,
                           frame2: GrayscaleImage
                           ) -> OpticalFlowImage:
    return OpticalFlowImage(cv2.calcOpticalFlowFarneback(
        frame1, frame2, flow=None, pyr_scale=.5, levels=3, winsize=9,
        iterations=1, poly_n=3, poly_sigma=1.1,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN))
