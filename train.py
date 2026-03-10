import warnings, os
warnings.filterwarnings('ignore')
import torch
from ultralytics_copy import RTDETR


if __name__ == '__main__':
    model = RTDETR('./wm-detr.yaml') 
    model.load('./weights/rtdetr-r50.pt') # loading pretrain weights 
    model.train(data='./datasets/data_RUOD.yaml', 
                cache=False, 
                imgsz=640, 
                epochs=120, 
                batch=64, # batch size workers=2, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0 
                seed=42, 
                warmup_epochs=3, 
                patience=30, 
                lr0=0.0005, # 降低初始学习率，防止不稳定 
                optimizer='AdamW', # 可选 'SGD'/'AdamW' 
                device='0,1,2,3', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案 
                project='runs/RUOD', 
                name='rtdetr-r50-RUOD-newf1-9', # save to project/name 
                )
