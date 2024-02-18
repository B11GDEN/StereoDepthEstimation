# StereoDepthEstimation

## Task
Задача Stereo Depth Estimation для робота дворецкого. Данных на домене лидарной сьемки внутри жилой квартиры я не нашел. 
Выбрал в качестве датасета KITTI2015 (https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) - задача
Stereo Depth Estimation дорожной ситуации.
При желании систему можно натренировать на домен внутри помещений.
## Repository Structure
Вся логика расположена в модуле stereodepth:
* datamodule - модуль для lightning datamodule
* losses - модуль для лосс функций
* metrics - модуль для метрик
* models - модуль для моделей

Систему можно масштабировать. Например добавляя новые лоссы, модели и т. д. в соответствующие модули.
## Installation
```bash
conda create -n stereo-depth python=3.10
conda activate stereo-depth
# Установить pytorch с официального сайта https://pytorch.org
pip install -r requirements.txt
```
## Train
Для тренировки используйте train.py. Подробнее о флагах смотри в parse_args
```bash
python3 train.py --batch-size 8 --max-epochs 100 --accelerator gpu --project sber-task --exp stereo_net
```
## Predict
Скрипт для инференса
```bash
# --left - путь до левого изображения
# --right - путь до правого изображения
# --model - путь до весов модели, уже обученные веса есть в папке weights
# --output - путь куда сохранится итоговая disparity map
python3 predict.py --left imgs/left.png --right imgs/right.png --model weights/stereo_net.pt --output result_disp.png
```
## Inference Example
Left Image
![Image alt](./imgs/left.png)
Right Image
![Image alt](./imgs/right.png)
Ground True Disparity
![Image alt](./imgs/disp.png)
Predicted Disparity
![Image alt](./imgs/color_disp.png)