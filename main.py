import os
import pickle as pkl
import shutil

import torch
import torch.nn as nn

from model.res_unet import Res_UNet
import utils.fid as lfid
import utils.util as U
from model.diffusion import Diffusion
from utils.gtmodel import InceptionV3
from utils.tfrecord import TFRDataloader

def train():
    denoizer.train()
    for idx, data in enumerate(loader):
        data = data.to(device)
        stat = diffusion.trainbatch(data)
        print(f'{idx // len(loader)}/{cfg["epoch"]} {idx % len(loader)}/{len(loader)} {stat["loss"]:.2}')
        if idx % 1000 == 0 :#and idx != 0:
            U.save_image(diffusion.sample(stride=cfg['stride'], embch=cfg['model']['embch'], x=xT),
                         f'{savefolder}/{idx}.jpg', s=0.5, m=0.5)


@torch.no_grad()
def check_fid(num_image):
    inception = InceptionV3([3]).to(device)
    with open('__celeba_real.pkl', 'rb') as f:
        realsigma, realmu = pkl.load(f)
        realsigma = realsigma.to(device)
        realmu = realmu.to(device)
    mvci = lfid.MeanCoVariance_iter(device)
    for idx in range(num_image // cfg['batchsize']):
        x = torch.randn(cfg['samplebatchsize'], cfg['model']['in_ch'], cfg['model']['size'], cfg['model']['size']).to(
            device)
        x = diffusion.sample(stride=cfg['stride'], embch=cfg['model']['embch'], x=x)
        mvci.iter(inception(x)[0])
    fid = lfid.fid(realsigma, realmu, *mvci.get(isbias=True))
    print(fid)


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
    if cfg['epoch']==-1:
        cfg['epoch']=int(500000/202589*cfg['batchsize'])
    if cfg['loss'] == 'mse':
        criterion = nn.MSELoss()
    denoizer = Res_UNet(**cfg['model']).to(device)
    loader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=cfg['epoch'], batch=cfg['batchsize'],
                           size=cfg['model']['size'], s=0.5, m=0.5)
    diffusion = Diffusion(denoizer=denoizer, criterion=criterion, device=device, **cfg['diffusion'])
    xT = torch.randn(cfg['samplebatchsize'], cfg['model']['in_ch'], cfg['model']['size'], cfg['model']['size']).to(
        device)
    train()
    # check_fid(2000)
