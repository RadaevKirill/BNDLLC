from typing import Optional

import cv2

from src.decoders.decoder import Decoder
from src.utils.models import FrameRGB


class OpenCVDecoder(Decoder):
    def __init__(self, video_path: str):
        self._video_path = video_path

    def run(self) -> Optional[FrameRGB]:
        cap = cv2.VideoCapture(self._video_path)

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            yield frame
