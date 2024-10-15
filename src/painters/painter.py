from abc import ABC, abstractmethod

from src.utils.models import FrameRGB, DetectionResults


class Painter(ABC):

    @abstractmethod
    def run(self, frame: FrameRGB, detections: DetectionResults) -> FrameRGB:
        raise NotImplementedError
