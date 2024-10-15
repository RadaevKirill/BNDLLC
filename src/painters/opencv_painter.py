from typing import Tuple

from src.painters.painter import Painter
from src.utils.models import FrameRGB, DetectionResults

import cv2


class OpenCVPainter(Painter):
    def __init__(self, box_color: Tuple[int, int, int] = (0, 255, 0), label_color: Tuple[int, int, int] = (0, 255, 0)):
        self._box_color = box_color
        self._label_color = label_color

    def run(self, frame: FrameRGB, detections: DetectionResults) -> FrameRGB:
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection.bbox)
            score = detection.score

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self._box_color, 5)
            frame = cv2.putText(frame, f'score: {score:.4f}', (x1, y1), 2, 2, self._label_color)

        return frame
