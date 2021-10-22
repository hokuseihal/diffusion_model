import os
import pickle as pkl
import shutil

import cloudpickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

from dataset.ucf101 import UCF101Dataset
from model.diffusion import Diffusion
from model.res_unet import Res_UNet
from plotter import Plotter
from utils import dataset as UD
from utils import fid as lfid
from utils.gtmodel import fid_inception_v3
from utils.tfrecord import TFRDataloader


def train():
    for data in loader:
        global gidx
        gidx += 1
        stat = diffusion.trainbatch(data, gidx)
        print(f'{epoch}/{cfg["epoch"]} {gidx % len(loader)}/{len(loader)} {stat["loss"]:.2}')
        if gidx % 2000 == 0:
            for stride in cfg['stride']:
                # U.save_image(diffusion.sample(stride=stride, embch=cfg['model']['embch'], x=xT),
                #              f'{savefolder}/{gidx}_{stride}.jpg', s=0.5, m=0.5)
                videoout = diffusion.sample(stride=stride, embch=cfg['model']['embch'], x=xT)
                for b in range(cfg['samplebatchsize']):
                    UD.writevideo(videoout[b], f'{savefolder}/{gidx}_{stride}_{b}.mp4', fps)
            if (cfg['fid']):
                fid = check_fid(2000)
                pltr.addvalue({'fid': fid}, gidx)
            with open(f'{savefolder}/model.cpkl', 'wb') as f:
                cloudpickle.dump({'model': diffusion.state_sict(), 'cfg': cfg}, f)


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
    parser.add_argument('--model', default='model/config/resunet3d.yaml')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--datasetpath', default='../data/')
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--loadckpt', default=False, action='store_true')
    args = parser.parse_args()

    savefolder = f'result/{args.savefolder}'
    device = args.device
    if args.loadckpt:
        with open(f'{savefolder}/ckpt', 'rb') as f:
            ckpt = cloudpickle.load(f)
            cfg = ckpt['cfg']
            state_dict = ckpt['model']
        denoizer = Res_UNet(**cfg['model']).to(device)
        denoizer.load_state_dict(state_dict)
    else:
        with open(args.model) as file:
            cfg = yaml.safe_load(file)
        os.makedirs('result', exist_ok=True)
        shutil.rmtree(savefolder, ignore_errors=True)
        os.mkdir(savefolder)
        denoizer = Res_UNet(**cfg['model']).to(device)
    if cfg['loss'] == 'mse':
        criterion = nn.MSELoss()
    if device == 'cuda':
        denoizer = torch.nn.DataParallel(denoizer)
    iscls = False
    numcls = None
    if cfg['dataset'] == 'celeba':
        loader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord',
                               batch=cfg['batchsize'] // cfg['diffusion']['subdivision'],
                               size=cfg['model']['size'], s=0.5, m=0.5)
        numimg = 202589
    elif cfg['dataset'] == 'stl10':
        loader = torch.utils.data.DataLoader(
            torchvision.datasets.STL10('../data/', transform=T.Compose(
                [T.Resize(cfg['model']['size']), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
                                       download=True), num_workers=4, batch_size=cfg['batchsize'])
        iscls = True
        numcls = 10
        numimg = 157 * 32
    elif cfg['dataset'] == 'ucf101':
        loader = torch.utils.data.DataLoader(
            UCF101Dataset('../data/ucf101/*avi', cfg['numframe'], cfg['videostride'], cfg['model']['size']),
            num_workers=4, batch_size=cfg['batchsize'])
        iscls = False
        numcls = None
        numimg = 13321
        fps = 25 // cfg['videostride']
    if cfg['epoch'] == -1:
        cfg['epoch'] = int(500000 / numimg * cfg['batchsize']) * cfg['diffusion']['subdivision']
    diffusion = Diffusion(denoizer=denoizer, criterion=criterion, device=device, iscls=iscls, numcls=numcls,
                          **cfg['diffusion'])
    xT = torch.randn(cfg['samplebatchsize'], cfg['model']['in_ch'], cfg['numframe'], cfg['model']['size'],
                     cfg['model']['size']).to(
        device)
    inception = fid_inception_v3().to(device)
    with open('celeba_real.pkl', 'rb') as f:
        realsigma, realmu = pkl.load(f)
        realsigma = realsigma.to(device)
        realmu = realmu.to(device)
    pltr = Plotter(f'{savefolder}/graph.jpg')

    gidx = 0
    for epoch in range(cfg['epoch']):
        train()
