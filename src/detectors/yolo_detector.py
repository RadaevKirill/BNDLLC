import logging
from typing import Dict

from ultralytics import YOLO

from src.detectors.detector import Detector
from src.utils.models import FrameRGB, DetectionResults, DetectionResult


class YoloDetector(Detector):
    def __init__(self, model_path: str):
        self._model_path = model_path

        self._model = YOLO(self._model_path)

    def run(self, frame: FrameRGB) -> DetectionResults:
        detection_results = []

        results = self._model(frame, verbose=False)[0]

        bboxes = results.boxes.xyxy.numpy()
        scores = results.boxes.conf.numpy()
        classes = results.boxes.cls.numpy()

        for b, s, c in zip(bboxes, scores, classes):
            if c:  # if class != 0, because 0 is person
                continue
            detection_results.append(DetectionResult(label=c, score=s, bbox=b))

        logging.info(f'preprocess: {results.speed['preprocess']}')
        logging.info(f'inference: {results.speed['inference']}')
        logging.info(f'postprocess: {results.speed['postprocess']}')

        return detection_results

