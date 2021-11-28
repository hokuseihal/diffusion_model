from multiprocessing import cpu_count

import torch
import torch.nn as nn
import yaml

from model.vae import AutoEncoder as Model
from utils.tfrecord import TFRDataloader


def operate(phase):
    if phase == 'train':
        model.train()
        loader = trainloader
    else:
        model.eval()
        loader = valloader
    for idx, (data, target) in enumerate(loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cfg', default='model/config/vae.yaml')
    parser.add_argument('--savefolder', default='tmp')
    args = parser.parse_args()

    savefolder=f'result/{args.savefolder}'
    with open(f'{savefolder}/cfg.yaml') as file:
        cfg = yaml.safe_load(file)
    if cfg['dataset'] == 'celeba':
        loader = TFRDataloader(path=args.datasetpath + '/ffhq.tfrecord',
                               batch=cfg['batchsize'] // cfg['subdivision'],
                               size=cfg['size'], s=0.5, m=0.5)
    device = args.device
    criterion = nn.MSELoss()
    model = Model()
    optimizer = torch.optim.Adam(model.parameters())

    for e in range(args.epoch):
        operate('train')
        operate('val')
