from abc import ABC, abstractmethod

from src.utils.models import FrameRGB


class Saver(ABC):

    @abstractmethod
    def run(self, frame: FrameRGB) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
