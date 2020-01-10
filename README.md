# FCN benchmark
This repo is a simple implementation of FCN. Technically, there remains lots of features to be fullfilled before it can be used by other people. It is just used for my experiments for now. 

## requirements
```
- torch
- torchvision
- cvtorch
```

## install
```bash
$ git clone https://github.com/iHateTa11B0y/fcns.git
$ python setup.py build develop
```

## training
```bash
$ python -m torch.distributed.launch tools/train.py --config-file "configs/my_yaml.yaml"
```

## eval and inference
```bash
$ python -m torch.distributed.launch tools/eval.py --config-file "configs/my_yaml.yaml"
```
```bash
$ python tools/infer.py --wts weights/model_epoch_6.pth --segm weights/segm.pth --gt /core1/data/home/niuwenhao/data/tiny_bg.json a
$ python tools/infer.py --wts weights/model_epoch_6.pth <img_url>
```
