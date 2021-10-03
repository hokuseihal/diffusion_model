import pickle as pkl

import torch
import torchvision.utils as tvu

from utils.fid import MeanCoVariance_iter
from utils.gtmodel import fid_inception_v3
from utils.tfrecord import TFRDataloader


def save_image(x, path, s=1, m=0):
    tvu.save_image(x * s + m, path)


def make_gt_inception(model, loader, device='cuda'):
    print('make inception output...')
    ret = []
    model = model.to(device)
    MCVI = MeanCoVariance_iter(device)
    for i, data in enumerate(loader):
        with torch.set_grad_enabled(False):
            print(f'\r{i},{len(loader)},{i / len(loader) * 100:2.0f}%', end='')
            img = data
            img = img.to(device)
            # print(img.shape)
            output = model(img)
            MCVI.iter(output)
            ret.append(output)

    # ret = torch.cat(ret, dim=0)
    # ret = ret.reshape(-1, 2048)
    # scov, sm = torch.from_numpy(np.cov(ret.cpu(), rowvar=False)), ret.mean(dim=0)
    # mcov, mm = MCVI.get(isbias=False)
    # bcov, bmm = MCVI.get(isbias=True)
    # return scov,sm
    return MCVI.get(isbias=True)


def make_fid_pkl(device='cuda'):
    print('make real stats')
    loader = TFRDataloader(path='../data/celeba.tfrecord', epoch=1, batch=128,
                           size=299, s=0.5, m=0.5)
    inception = fid_inception_v3().to(device)
    realsigma, realmu = make_gt_inception(inception, loader, device)
    with open('celeba_real.pkl', 'wb') as f:
        pkl.dump([realsigma.cpu(), realmu.cpu()], f)


class iterrepeater:
    def __init__(self):
        self.itr = iter(range(5))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.itr)
        except StopIteration:
            self.itr = iter(range(5))
            raise StopIteration


if __name__ == '__main__':
    # makefidpkl=True
    # if(makefidpkl):
    #     make_fid_pkl()
    I = iterrepeater()
    for e in range(4):
        for i in I:
            print(e,i)
