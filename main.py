import os
import pickle as pkl
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.fid as lfid
import utils.util as U
from model.diffusion import Diffusion
from model.res_unet import Res_UNet
from plotter import Plotter
from utils.gtmodel import fid_inception_v3
from utils.tfrecord import TFRDataloader


def train():
    denoizer.train()
    for idx, data in enumerate(loader):
        stat = diffusion.trainbatch(data, idx)
        print(f'{idx // len(loader)}/{cfg["epoch"]} {idx % len(loader)}/{len(loader)} {stat["loss"]:.2}')
        if idx % 500 == 0 and idx != 0:
            U.save_image(diffusion.sample(stride=cfg['stride'], embch=cfg['model']['embch'], x=xT),
                         f'{savefolder}/{idx}.jpg', s=0.5, m=0.5)
            if (cfg['fid']):
                fid = check_fid(253)
                pltr.addvalue({'fid': fid}, idx)


@torch.no_grad()
def check_fid(num_image):
    mvci = lfid.MeanCoVariance_iter(device)
    for idx in range(num_image // cfg['batchsize'] + 1):
        print(idx, num_image, cfg['batchsize'])
        x = torch.randn(cfg['samplebatchsize'], cfg['model']['in_ch'], cfg['model']['size'], cfg['model']['size']).to(
            device)
        x = diffusion.sample(stride=cfg['stride'], embch=cfg['model']['embch'], x=x)
        x = F.interpolate(x, (299, 299))
        mvci.iter(inception(x))
    fid = lfid.fid(realsigma, realmu, *mvci.get(isbias=True))
    print(f'{fid=}')
    return fid


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
    if cfg['epoch'] == -1:
        cfg['epoch'] = int(500000 / 202589 * cfg['batchsize'])*cfg['diffusion']['subdivision']
    if cfg['loss'] == 'mse':
        criterion = nn.MSELoss()
    denoizer = Res_UNet(**cfg['model']).to(device)
    loader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=cfg['epoch'],
                           batch=cfg['batchsize'] // cfg['diffusion']['subdivision'],
                           size=cfg['model']['size'], s=0.5, m=0.5)
    diffusion = Diffusion(denoizer=denoizer, criterion=criterion, device=device, **cfg['diffusion'])
    xT = torch.randn(cfg['samplebatchsize'], cfg['model']['in_ch'], cfg['model']['size'], cfg['model']['size']).to(
        device)
    inception = fid_inception_v3().to(device)
    with open('celeba_real.pkl', 'rb') as f:
        realsigma, realmu = pkl.load(f)
        realsigma = realsigma.to(device)
        realmu = realmu.to(device)
    pltr = Plotter(f'{savefolder}/graph.jpg')

    train()
