from multiprocessing import cpu_count

import torch
import torch.nn as nn
import yaml

from model.vae import AutoEncoder as Model
from utils.tfrecord import TFRDataloader
from torch.cuda.amp import autocast,GradScaler
def operate(phase):
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader
    for idx, data in enumerate(loader):
        data = data.to(device)
        with autocast():
            output = model.img2img(data,grad_enc=True)
            loss = criterion(output, data)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        print(f'{phase}:{idx}/{len(loader)}:{loss.item():.4f}')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cfg', default='model/config/vae.yaml')
    parser.add_argument('--savefolder', default='tmp')
    parser.add_argument('--datasetpath', default='../data/')
    args = parser.parse_args()

    savefolder=f'result/{args.savefolder}'
    with open(args.cfg) as file:
        cfg = yaml.safe_load(file)
    if cfg['dataset'] == 'celeba':
        trainloader = TFRDataloader(path=args.datasetpath + '/ffhq_train.tfrecord',
                               batch=cfg['batchsize'] // cfg['subdivision'],
                               size=cfg['size'], s=0.5, m=0.5)
        valloader = TFRDataloader(path=args.datasetpath + '/ffhq_val.tfrecord',
                               batch=cfg['batchsize'] // cfg['subdivision'],
                               size=cfg['size'], s=0.5, m=0.5)

    device = args.device
    criterion = nn.L1Loss()
    model = Model(**cfg['vae']).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scaler=GradScaler()

    for e in range(args.epoch):
        operate('train')
        operate('val')
