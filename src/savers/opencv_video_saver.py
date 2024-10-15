from pathlib import Path
from typing import Tuple

import cv2

from src.savers.saver import Saver
from src.utils.models import FrameRGB


class OpenCVVideoSaver(Saver):
    def __init__(self, target_file: str, fps: int, width: int, height: int):
        self._target_file = str(target_file)
        self._fps = fps
        self._frame_size = (width, height)

        self._saver = cv2.VideoWriter(self._target_file, cv2.VideoWriter_fourcc(*'mp4v'), self._fps, self._frame_size)

    def run(self, frame: FrameRGB) -> None:
        self._saver.write(frame)

    def stop(self) -> None:
        self._saver.release()
