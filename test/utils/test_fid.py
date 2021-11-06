import pickle as pkl

import torch
import torch.nn.functional as F
from utils import fid as lfid
from utils.fid import MeanCoVariance_iter as MCVI
from utils.gtmodel import fid_inception_v3
from utils.tfrecord import TFRDataloader


def fid_test(numfid=10000):
    batchsize=64
    size=128
    loader = TFRDataloader(path='../data/celeba.tfrecord', batch=batchsize, size=size, s=0.5, m=0.5)
    inception = fid_inception_v3().cuda()
    mcvi = MCVI('cuda')
    with open('celeba_real.pkl', 'rb') as f:
        realsigma, realmu = pkl.load(f)
        realsigma = realsigma.cuda()
        realmu = realmu.cuda()

    for i, data in enumerate(loader):
        with torch.set_grad_enabled(False):
            print(f'\r{i*batchsize},{i},{len(loader)},{i / len(loader) * 100:2.0f}%', end='')
            output = inception(F.interpolate(data.cuda(),(299,299)))
            mcvi.iter(output)
        if i*batchsize>numfid:break

    fid = lfid.fid(realsigma, realmu, *mcvi.get(isbias=True))
    print(numfid,fid)
    # assert fid < 1e-3,fid

if __name__=='__main__':
    for numfid in [50,100,500,1000,5000,10000,50000,100000]:
        test_fid(numfid=numfid)