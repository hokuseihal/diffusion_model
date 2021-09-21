import torch
import torch.nn as nn


class AdaGN(nn.GroupNorm):

    def forward(self, x, ys, yb):
        '''
        :param x: input
        :param ys: time embedding
        :param yb: class embedding
        :return:
        '''
        B,C,H,W=x.shape
        return (1 + ys) * super().forward(x) + yb


if __name__ == '__main__':
    m = AdaGN(3, 3)
    data = torch.randn(8, 3, 16, 16)
    out = m(data, torch.randn(8).view(8, -1, 1, 1),torch.randn(8).view(8,-1,1,1))
    print(out.shape)
