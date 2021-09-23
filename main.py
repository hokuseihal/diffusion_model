import os
import shutil

import torch
import torch.nn as nn
from torchvision.utils import save_image

import model
from model.diffusion import Diffusion
from utils.tfrecord import TFRDataloader


def train():
    denoizer.train()
    for idx, data in enumerate(loader):
        data = data.to(device)
        stat = diffusion.trainbatch(data)
        print(f'{idx//len(loader)}/{cfg["epoch"]} {idx%len(loader)}/{len(loader)} {stat["loss"]:.2}')
        if idx % 1000 == 0:
            save_image(diffusion.sample(stride=cfg['stride'],
                                        shape=(cfg['samplebatchsize'], 3, cfg['model']['size'], cfg['model']['size']),embch=cfg['model']['embch']),
                       f'{savefolder}/{idx}.jpg')


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model/config/resunet.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--datasetpath', default='../data/')
    parser.add_argument('--savefolder', default='tmp')
    args = parser.parse_args()

    with open(args.model) as file:
        cfg = yaml.safe_load(file)
    device = args.device
    os.makedirs('result', exist_ok=True)
    savefolder = f'result/{args.savefolder}'
    shutil.rmtree(savefolder, ignore_errors=True)
    os.mkdir(savefolder)
    if cfg['loss'] == 'mse':
        criterion = nn.MSELoss()
    denoizer = model.Res_UNet(**cfg['model']).to(device)
    loader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=cfg['epoch'], batch=cfg['batchsize'],
                           size=cfg['model']['size'], s=0.5, m=0.5)
    diffusion = Diffusion(denoizer=denoizer, criterion=criterion, schedule=cfg['schedule'],
                          device=device,lr=cfg['lr'],eta=cfg['eta'])
    train()
    # check_fid()
