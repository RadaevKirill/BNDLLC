from abc import ABC, abstractmethod

from src.utils.models import FrameRGB
from typing import Iterator


class Decoder(ABC):

    @abstractmethod
    def run(self) -> Iterator[FrameRGB]:
        raise NotImplementedError