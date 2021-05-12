from typing import List

from dataclasses import dataclass

from paitypes.image import BGRImage


@dataclass
class InferenceRequest:
    images: List[BGRImage]
    require_poses: bool = True
    require_detections: bool = True
