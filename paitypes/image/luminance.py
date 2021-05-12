import numpy as np
from .color import BGRImage
from paitypes.image import BGRImage, bgr_image_to_ycrcb


def bgr_image_to_luminance(frame: BGRImage) -> float:
    ycb = bgr_image_to_ycrcb(frame)
    y_channel = ycb[:, :, 0].astype(np.float64)

    m = y_channel.mean() / 255.0
    return m
