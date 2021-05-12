import time
import argparse
from Trainer import Trainer
from models import SPnet,U_net_Spec
from losses import TotalLoss
from loaders import Loader
from utils import Utilizer
import torch.nn as nn
from retinex import MSRCR,config
from tqdm import tqdm
import scipy.io as scio
import numpy as np
import torchvision.transforms as tf
import torch
from torchvision.utils import save_image
import os
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=200)
    parser.add_argument('--initLR',type=float,default=0.001)
    parser.add_argument('--step_size',type=int,default=100)
    parser.add_argument('--decray_weight',type=float,default=0.5)
    parser.add_argument('--device_id',type=list,default=[4,5,6,7])
    parser.add_argument('--device',type=str,default='cuda:4')
    parser.add_argument('--Time',type=str,default='-'.join(map(lambda i:str(i),time.localtime(time.time())[0:5]))+'/')
    parser.add_argument('--checkpoint',type=str,default='checkpoint/')
    parser.add_argument('--hdrfile',type=str,default='./1.hdr')
    parser.add_argument('--ncolor',type=int,default=128)
    parser.add_argument('--expendfile',type=str,default='downsample')
    parser.add_argument('--root',type=str,default='./Data/')
    parser.add_argument('--weights',type=list,default=[10,1,0.01,1])
    parser.add_argument('--train_size',type=float,default=0.9)
    opts=parser.parse_args()


    utilizer=Utilizer(opt=opts)

    loader=Loader(opts)

    model=SPnet(opts=opts)

    criterion=TotalLoss(opts)

    trainer=Trainer(opts=opts,loader=loader,model=model,loss=criterion,utilizer=utilizer)
#     trainer.eval(999)
    trainer.train()


