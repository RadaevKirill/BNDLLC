from statistics import mean

import hydra
from omegaconf import DictConfig

from src.decoders.decoder import Decoder
from src.detectors.detector import Detector
from src.savers.saver import Saver
from src.painters.painter import Painter


class Runner:
    def __init__(self, decoder: Decoder, detector: Detector, painter: Painter, saver: Saver):
        self._decoder = decoder
        self._detector = detector
        self._painter = painter
        self._saver = saver

    def run(self) -> None:
        for frame in self._decoder.run():
            detections = self._detector.run(frame)
            frame = self._painter.run(frame, detections)
            self._saver.run(frame)

        self._saver.stop()

    def metric_calculate(self) -> None:
        preproc_time = []
        infer_time = []
        postproc_time = []

        with open(f'{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}/__main__.log', 'r') as file:
            logs = file.readlines()

        for log in logs:
            info = log.split('- ')[-1].replace('\n', '').split(':')
            if info[0] == 'preprocess':
                preproc_time.append(float(info[1]))
            if info[0] == 'inference':
                infer_time.append(float(info[1]))
            if info[0] == 'postprocess':
                postproc_time.append(float(info[1]))

        print(f'Mean latency preprocess: {mean(preproc_time):04f}')
        print(f'Mean latency infer: {mean(infer_time):04f}')
        print(f'Mean latency postprocess: {mean(postproc_time):04f}')


@hydra.main(config_path='../assets', config_name='config', version_base='1.3.2')
def main(config: DictConfig) -> None:
    runer = hydra.utils.instantiate(config['runner'])
    runer.run()
    runer.metric_calculate()


if __name__ == '__main__':
    main()
