from dataclasses import dataclass
from typing import List

from nptyping import NDArray, Shape, UInt8, Float32

FrameRGB = NDArray[Shape['* height, * width, 3 rgb'], UInt8]
OpenVINOInput = NDArray[Shape['1, 3, 512, 512'], UInt8]
OpenVINOOutput = NDArray[Shape['1, 1, 200, 7'], Float32]

@dataclass
class DetectionResult:
    label: int
    score: float
    bbox: List[int]  # x1 y1 x2 y2


DetectionResults = List[DetectionResult]
