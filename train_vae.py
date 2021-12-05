import os

import torch
import torch.nn as nn
import torchvision.transforms as T
import yaml
from torch.cuda.amp import autocast, GradScaler

import utils.util as U
import wandb
from core import Plotter
from model.vae import AutoEncoder
from model.res_unet import Res_UNet
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
                if cfg['model']['moco']['flag']:
                    loss=model.trainmoco(data,device)
                else:
                    loss, outimg = model.trainenc_dec(data, criterion, device)
            if phase == 'train':
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif not cfg['model']['moco']['flag']:
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

    savefolder = f'result/{args.savefolder}'
    os.makedirs(savefolder, exist_ok=True)
    with open(args.cfg) as file:
        cfg = yaml.safe_load(file)
    if not args.dis_wandb:
        wandb.init(project='vae' if cfg['model']['moco']['flag'] else 'moco')
        wandb.run.name = args.savefolder
    if cfg['dataset'] == 'celeba':
        trainloader = TFRDataloader(path=args.datasetpath + '/ffhq_train.tfrecord',
                                    batch=cfg['batchsize'],
                                    size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['model']['moco']['flag'])
        valloader = TFRDataloader(path=args.datasetpath + '/ffhq_val.tfrecord',
                                  batch=cfg['batchsize'],
                                  size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['model']['moco']['flag'])
        trainloader = TFRDataloader(path='tmp.tfrecord',
                                    batch=cfg['batchsize'],
                                    size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['model']['moco']['flag'])
        valloader = TFRDataloader(path='tmp.tfrecord',
                                  batch=cfg['batchsize'],
                                  size=cfg['size'], s=0.5, m=0.5, get_aug=cfg['model']['moco']['flag'])

    device = args.device
    criterion = nn.L1Loss()
    if 'vae.yaml' in args.cfg:
        Model=AutoEncoder
    elif 'resunet.yaml' in args.cfg:
        Model=Res_UNet
    model = Model(**cfg['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    scaler = GradScaler()

    gidx = -1
    for e in range(args.epoch):
        operate('train')
        operate('val')
        torch.save(model.state_dict(),f'{savefolder}/model.pth')
    if not args.dis_wandb:
        wandb.finish()
