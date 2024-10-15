from abc import ABC, abstractmethod

from src.utils.models import FrameRGB, DetectionResults
from typing import Iterator


class Detector(ABC):

    @abstractmethod
    def run(self, frame: FrameRGB) -> DetectionResults:
        raise NotImplementedError