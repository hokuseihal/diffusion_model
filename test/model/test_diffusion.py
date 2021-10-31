import torch
import yaml

from model.diffusion import Diffusion
from model.res_unet import Res_UNet


def test_gentlenextsample(size=(8, 3, 8, 8), e_thres=1e-7):
    with open('model/config/resunet.yaml') as f:
        cfg = yaml.safe_load(f)
    denoizer = Res_UNet(**cfg['model'])
    diffusion = Diffusion(denoizer=denoizer, criterion=None, device='cpu', iscls=None, numcls=None, **cfg['diffusion'])
    x0 = torch.randn(size)
    e = torch.randn(size)
    for t, stride in [(1, 1), (4, 1), (500, 1), (999, 1), (2, 2), (4, 2), (500, 2), (999, 2), (4, 4), (500, 4),
                      (999, 4), (100, 100), (999, 100)]:
        assert t - stride >= 0
        xt_1 = diffusion.a[t - stride].sqrt() * x0 + (1 - diffusion.a[t - stride]).sqrt() * e
        xt_x = diffusion.a[t].sqrt() * x0
        xt_e = (1 - diffusion.a[t]).sqrt() * e
        assert (xt_1 - diffusion.gentlenextsample(xt_x, xt_e, t, stride)).mean() < e_thres
