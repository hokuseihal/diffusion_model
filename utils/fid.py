import torch
from scipy import linalg


class MeanCoVariance_iter:
    def __init__(self, device):
        self.n = 0
        self.s = 0
        self.mu = torch.zeros(1).to(device)

    def iter(self, xn):
        xn = xn.double()
        B, C= xn.shape
        # B, C, _, _ = xn.shape
        # xn = xn.view(B, C)
        mun = (self.n * self.mu + xn.sum(dim=0)) / (self.n + B)
        sn = 1 / (self.n + B) * torch.einsum('ij,ik->jk', xn, xn) \
             + self.n / (self.n + B) * torch.einsum('i,j->ij', self.mu, self.mu) \
             - torch.einsum('i,j->ij', mun, mun) \
             + self.n / (self.n + B) * self.s
        self.s = sn
        self.n += B
        self.mu = mun

    def get(self, isbias=True):
        bias = self.n / (self.n - 1) if isbias else 1
        return self.s * bias, self.mu


def sqrtm(x):
    m = x.detach().cpu().numpy()
    sqrtm, _ = linalg.sqrtm(m, disp=False)
    return torch.from_numpy(sqrtm.real).to(x.device)


def make_gt_inception(model, loader, device):
    print('make inception output...')
    ret = []
    model = model.to(device)
    MCVI = MeanCoVariance_iter(device)
    for i, data in enumerate(loader):
        with torch.set_grad_enabled(False):
            print(f'\r{i},{len(loader)},{i / len(loader) * 100:2.0f}%', end='')
            img = data[1]
            img = img.to(device)
            # print(img.shape)
            output = model(img)[0]
            MCVI.iter(output)
            ret.append(output)

    return MCVI.get(isbias=True)


def fid(cogtsigma, gtmean, cofakesigma, fakemean):
    return (torch.norm(gtmean - fakemean) ** 2 \
            + torch.trace(cogtsigma) + torch.trace(cofakesigma) - 2 * (
                torch.trace(sqrtm(cogtsigma @ cofakesigma)))).item()
