import torch
import torch.nn as nn

import model
from model.diffusion import Diffusion
from utils.tfrecord import TFRDataloader
from torchvision.utils import save_image

import os
import shutil

def train():
    denoizer.train()
    for idx, data in enumerate(loader):
        data = data.to(device)
        stat = diffusion.trainbatch(data)
        print(f'{e}/{args.epoch} {idx}/{len(loader)} {stat["loss"]:.2}')
        if idx%100==0:
            save_image(diffusion.sample(n_sample_iter=args.n_iter,shape=(args.batchsize,3,args.size,args.size)),f'{savefolder}/{e}_{idx}.jpg')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', default=32, type=int)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--model', default='unetsimple')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--type', default='ddpm')
    parser.add_argument('--n_iter', default=1000, type=int)
    parser.add_argument('--schedule', default='linear')
    parser.add_argument('--datasetpath',default='../data/')
    parser.add_argument('--size',default=128,type=int)
    parser.add_argument('--savefolder',default='tmp')
    args = parser.parse_args()

    device = args.device
    os.makedirs('result',exist_ok=True)
    savefolder=f'result/{args.savefolder}'
    shutil.rmtree(savefolder,ignore_errors=True)
    os.mkdir(savefolder)
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    if args.model == 'unetsimple':
        denoizer = model.UNet(is_time_embed=True).to(device)
    optimizer = torch.optim.Adam
    loader = TFRDataloader(path=args.datasetpath + '/celeba.tfrecord', epoch=args.epoch, batch=args.batchsize,
                           size=args.size, s=0.5, m=0.5)
    diffusion = Diffusion(denoizer=denoizer, criterion=criterion, optimizer=optimizer, schedule=args.schedule, device=device)
    for e in range(args.epoch):
        train()
        # check_fid()
