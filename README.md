# Covid-Detection-Radiographs

In this competition we are given the set with COVID patients' radiographs. The task is to detect suspicious areas of possible lung damage and classify images into 4 different classes: "negative", "typical", "indeterminate", "atypical".

Detection part is implemented in `./detection`. 
Faster-RCNN model is used as a backbone and is specified in `./detection/models.py`. You can also run `./detection/validate.py` to measure MAP metric. Before training the model, you have to change parameters in `./detection/config.py` and then run:
``` python train.py --device={your_device} --fold={fold in cv} --debug={flag to debug} ```
The model


Classification part is almost the same. Implemeted backbones are resnet, efficientnet or swin-transformer, however, any model from timm library can be chosen with slight modifications in `./classification/models.py` module.

Pre-commit hook usage
Steps:
```
pip install pre-commit
```
verify you have `.pre-commit-config.yaml` file
```
pre-commit install
```
