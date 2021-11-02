import os
import pickle as pkl
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import utils.fid as lfid
import utils.util as U
import wandb
from model.diffusion import Diffusion
from model.res_unet import Res_UNet
from plotter import Plotter
from utils.gtmodel import fid_inception_v3
from utils.tfrecord import TFRDataloader


def train():
    global gidx
    for idx, data in enumerate(loader):
        gidx += 1
        stat = diffusion.trainbatch(data, gidx)
        print(f'{epoch}/{cfg["epoch"]} {gidx % len(loader)}/{len(loader)} {stat["loss"]:.2}')
        if use_wandb: wandb.log(stat)
        if idx % 2000 == 0:
            for stride in cfg['stride']:
                gen_img = U.make_grid(diffusion.sample(stride=stride, embch=cfg['model']['embch'], x=xT), s=0.5, m=0.5)
                if use_wandb:
                    wandb.log({'output': wandb.Image(gen_img, caption=f'{gidx}_{stride}')})
                else:
                    U.save_image(gen_img, f'{savefolder}/{gidx}_{stride}.jpg', s=0.5, m=0.5)
            if (cfg['fid']):
                fid = check_fid(2000)
                pltr.addvalue({'fid': fid}, gidx)
    torch.save(denoizer.module.state_dict(), f'{savefolder}/model.pth')
    with open(f'{savefolder}/epoch.txt', 'w') as f:
        f.write(f'{epoch},{gidx}')


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
    parser.add_argument('--savefolderbase', default='.')
    parser.add_argument('--changecfg', default='')
    parser.add_argument('--dis_wandb', default=False, action='store_true')
    parser.add_argument('--restart', default=False, action='store_true')
    args = parser.parse_args()

    savefolder = f'{args.savefolderbase}/result/{args.savefolder}'
    device = args.device
    use_wandb=not args.dis_wandb
    if not args.restart:
        os.makedirs(f'{args.savefolderbase}/result', exist_ok=True)
        shutil.rmtree(savefolder, ignore_errors=True)
        os.mkdir(savefolder)
        shutil.copy(args.model, f'{savefolder}/cfg.yaml')
    with open(f'{savefolder}/cfg.yaml') as file:
        cfg = yaml.safe_load(file)
    cfg = U.setcfg(cfg, args.changecfg)
    with open(f'{savefolder}/cfg.yaml', 'w') as f:
        yaml.dump(cfg, f)
    denoizer = Res_UNet(**cfg['model']).to(device)
    startepoch = 0
    gidx = 0
    if args.restart:
        denoizer.load_state_dict(torch.load(f'{savefolder}/model.pth'))
        with open(f'{savefolder}/epoch.txt') as f:
            startepoch, gidx = list(map(int, f.read().strip().split(',')))
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
    if cfg['epoch'] == -1:
        cfg['epoch'] = int(500000 / numimg * cfg['batchsize']) * cfg['diffusion']['subdivision']
    diffusion = Diffusion(denoizer=denoizer, criterion=criterion, device=device, iscls=iscls, numcls=numcls,
                          **cfg['diffusion'])
    xT = torch.randn(cfg['samplebatchsize'], cfg['model']['in_ch'], cfg['model']['size'], cfg['model']['size']).to(
        device)
    inception = fid_inception_v3().to(device)
    with open('celeba_real.pkl', 'rb') as f:
        realsigma, realmu = pkl.load(f)
        realsigma = realsigma.to(device)
        realmu = realmu.to(device)
    pltr = Plotter(f'{savefolder}/graph.jpg')
    if use_wandb:
        wandb.init(project='main')
        wandb.run.name = args.savefolder
        wandb.config = cfg
    for epoch in range(startepoch, cfg['epoch']):
        train()
    if use_wandb:
        wandb.finish()
