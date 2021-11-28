import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish


class GenerativeCell(nn.Module):
    def __init__(self, ch_in, ch_out, expand_rate, bntype, bn_eps, bn_momentum, se_r):
        super(GenerativeCell, self).__init__()
        self.layers = nn.Sequential(*[
            nn.BatchNorm2d(ch_in, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_in, ch_out*expand_rate, 1),
            bnswish(bntype)(ch_out*expand_rate, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_out*expand_rate, ch_out * expand_rate, 5, padding=2, groups=ch_out),
            bnswish(bntype)(ch_out*expand_rate, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_out*expand_rate, ch_out, 1),
            nn.BatchNorm2d(ch_out, eps=bn_eps, momentum=bn_momentum),
            SE(ch_out, se_r)
        ])

    def forward(self, x):
        return self.layers(x)


class GAP(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        return F.upsample(x, mode='bilinear', size=1).reshape(B, C)


class SE(nn.Module):
    def __init__(self, feature, r):
        super(SE, self).__init__()
        self.layers = nn.Sequential(*[
            GAP(),
            nn.Linear(feature, feature // r),
            nn.ReLU(inplace=True),
            nn.Linear(feature//r, feature),
            nn.Sigmoid()
        ])

    def forward(self, x):
        return self.layers(x)[...,None,None] * x


class BNSwish(nn.Module):
    def __init__(self, feature, **kwargs):
        super(BNSwish, self).__init__()
        self.layers = nn.Sequential(*[nn.BatchNorm2d(feature, **kwargs), nn.SiLU()])

    def forward(self, x):
        return self.layers(x)


def bnswish(type):
    return BNSwish if type == 'normal' else SyncBatchNormSwish


class EncoderCell(nn.Module):
    def __init__(self, ch_in, ch_out, bntype, bn_eps, bn_momentum, se_r, **kwargs):
        super(EncoderCell, self).__init__()
        self.layers = nn.Sequential(*[
            bnswish(bntype)(ch_in, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_in, ch_out, 3, padding=1),
            bnswish(bntype)(ch_out, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            SE(ch_out, se_r)
        ])

    def forward(self, x):
        return self.layers(x)


class Coder(nn.Module):
    def __init__(self, moduletype, feature, block_features, num_cell_per_block, in_ch, out_ch, **kwargs):
        super(Coder, self).__init__()
        module = EncoderCell if moduletype == 'encoder' else GenerativeCell
        layres = [nn.Conv2d(in_ch, feature * block_features[0], 1)]
        prefeature = feature
        block_features = [ch * feature for ch in block_features]
        for b_idx,feature in enumerate(block_features):
            layres.append(module(prefeature, feature, **kwargs))
            for idx in range(num_cell_per_block - 1):
                layres.append(module(feature, feature, **kwargs))
            layres.append(nn.Upsample(mode='bilinear', scale_factor=0.5 if moduletype == 'encoder' else 2))
            prefeature=feature
        layres.append(nn.Conv2d(block_features[-1], out_ch, 1))
        self.layers = nn.Sequential(*layres)

    def forward(self, x):
        return self.layers(x)


class AutoEncoder(nn.Module):
    '''
    Change [NVAE](https://github.com/NVlabs/NVAE) to non-hierarchical
    '''

    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = Coder('encoder', **kwargs)
        self.decoder = Coder('decoder', **kwargs)

    # def forward(self, x):
    #     raise ValueError('use other function')

    def load_encoder_weight(self, path):
        self.encoder.load_state_dict(torch.load(path))

    def img2latent(self, x):
        return self.encoder(x)

    def latent2img(self, z):
        return self.decoder(z)

    def img2img(self, x, grad_enc=True):
        with torch.set_grad_enabled(grad_enc):
            x = self.encoder(x)
        return self.decoder(x)


if __name__ == '__main__':
    m = AutoEncoder(
        feature=32,
        block_features=[1, 1, 2, 2, 4, 4],
        num_cell_per_block=2,
        bntype='normal',
        bn_eps=1e-5,
        bn_momentum=0.05,
        se_r=16,
        expand_rate=3,
        in_ch=3,
        out_ch=3
    )
    data = torch.randn(8, 3, 128, 128)
    output = m.img2img(data)
    print(output.shape)
    # from torch.utils.tensorboard import SummaryWriter
    # w=SummaryWriter()
    # w.add_graph(m,data)
    # w.close()
    # torch.onnx.export(m,data,'tmp.onnx',opset_version=11)
