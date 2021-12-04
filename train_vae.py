import os

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.cuda.amp import autocast, GradScaler

import utils.util as U
import wandb
from core import Plotter
from model.vae import AutoEncoder as Model
from utils.tfrecord import TFRDataloader


def operate(phase):
    global gidx
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader
    for idx, data in enumerate(loader):
        gidx += 1
        with torch.set_grad_enabled(phase == 'train'):
            with autocast():
                loss, outimg = model.trainenc_dec(data, criterion, device)
            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                gen_img = U.make_grid(torch.cat([data,outimg.to(torch.float32)],dim=2), s=0.5, m=0.5)
                if not args.dis_wandb:
                    wandb.log({'output': wandb.Image(T.ToPILImage()(gen_img), caption=f'{gidx}')})
                else:
                    U.save_image(gen_img, f'{savefolder}/{gidx}.jpg')
        optimizer.zero_grad()
        if not args.dis_wandb: wandb.log({f"loss_{phase}": loss.item()})

        print(f'{phase}:{idx}/{len(loader)}:{loss.item():.4f}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cfg', default='model/config/vae.yaml')
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--datasetpath', default='../data/')
    parser.add_argument('--dis_wandb', default=False, action='store_true')
    args = parser.parse_args()

    if not args.dis_wandb:
        wandb.init(project='main')
        wandb.run.name = args.savefolder
    savefolder = f'result/{args.savefolder}'
    os.makedirs(savefolder, exist_ok=True)
    with open(args.cfg) as file:
        cfg = yaml.safe_load(file)
    if cfg['dataset'] == 'celeba':
        trainloader = TFRDataloader(path=args.datasetpath + '/ffhq_train.tfrecord',
                                    batch=cfg['batchsize'] // cfg['subdivision'],
                                    size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['vae']['moco']['flag'])
        valloader = TFRDataloader(path=args.datasetpath + '/ffhq_val.tfrecord',
                                  batch=cfg['batchsize'] // cfg['subdivision'],
                                  size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['vae']['moco']['flag'])
        trainloader = TFRDataloader(path='tmp.tfrecord',
                                    batch=cfg['batchsize'] // cfg['subdivision'],
                                    size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['vae']['moco']['flag'])
        valloader = TFRDataloader(path='tmp.tfrecord',
                                  batch=cfg['batchsize'] // cfg['subdivision'],
                                  size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['vae']['moco']['flag'])

    device = args.device
    criterion = nn.L1Loss()
    model = Model(**cfg['vae']).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler()

    gidx = -1
    for e in range(args.epoch):
        operate('train')
        operate('val')
    if not args.dis_wandb:
        wandb.finish()
