import logging
from time import perf_counter

import cv2
import numpy as np
import openvino as ov

from src.detectors.detector import Detector
from src.utils.models import FrameRGB, DetectionResults, DetectionResult, OpenVINOOutput, OpenVINOInput


class OpenVINODetector(Detector):
    def __init__(self, model_path: str, width: int, height: int, score_threshold: float):
        self._core = ov.Core()
        self._model = self._core.compile_model(model_path, 'AUTO')
        self._infer_request = self._model.create_infer_request()

        self._input_layer = self._model.input(0)
        self._output_layer = self._model.output(0)

        self._width = width
        self._height = height
        self._score_threshold = score_threshold

    def run(self, frame: FrameRGB) -> DetectionResults:
        blob = self.__preprocessor(frame)
        detections = self.__infer(blob)
        return self.__postprocessor(detections)

    def __preprocessor(self, frame: FrameRGB) -> OpenVINOInput:
        N, C, H, W = self._input_layer.shape

        start = perf_counter()
        blob = np.expand_dims(np.transpose(cv2.resize(src=frame, dsize=(W, H)), (2, 0, 1)), 0).astype(np.float32)
        logging.info(f'preprocess: {perf_counter() - start}')

        return blob

    def __infer(self, blob: OpenVINOInput) -> OpenVINOOutput:
        start = perf_counter()
        result = self._model(blob)[self._output_layer]
        logging.info(f'inference: {perf_counter() - start}')

        return result

    def __postprocessor(self, detections: OpenVINOOutput) -> DetectionResults:
        # TODO Переписать данный код без использования циклов
        start = perf_counter()
        result = []
        for det in detections[0][0]:
            if det[2] < self._score_threshold:
                continue
            x1, x2 = det[3::2] * self._width
            y1, y2 = det[4::2] * self._height

            result.append(DetectionResult(label=det[1], score=det[2], bbox=[x1, y1, x2, y2]))

        logging.info(f'postprocess: {perf_counter() - start}')

        return result
