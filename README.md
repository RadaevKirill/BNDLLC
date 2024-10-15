# BNDLLC

### Данный проект преднозначен для сравнения двух моделей детекции.

В качестве моделей детекции были выбраны две модели:
- Person-detection-0202 (Из зоопарка OpenVINO)
- YoloV10-nano (Из зоопарка Ultralitics)

Более подробные результаты сравнения можно найти в файле ```assets/results.txt```

Настройка окружения: 
```shell
python3.12 -m venv venv
source venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

Сборка докер образа
```shell
docker build -f dev/Dockerfile -t bndllc .
```

Запуск тестового прогона на данных моделях выполняется с помощью следующих комманд:
```shell
# Для модели YOLOv10 nano
python -m src runner.detector='${yolo_v10_n}'

# Для модели Person detection 0202
python -m src runner.detector='${openvino_0202}'
```

В результате работы даного скрипта мы получим, визуализацию работы модлей на тестовом видео, которое будет находится в папке ```assets/video/```