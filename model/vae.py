import torch
import torch.nn as nn
import torch.nn.functional as F

from thirdparty.inplaced_sync_batchnorm import SyncBatchNormSwish


class GenerativeCell(nn.Module):
    def __init__(self, ch_in, ch_out, expand_rate, bntype, bn_eps, bn_momentum, se_r, **kwargs):
        super(GenerativeCell, self).__init__()
        self.layers = nn.Sequential(*[
            nn.BatchNorm2d(ch_in, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_in, ch_out * expand_rate, 1),
            bnswish(bntype)(ch_out * expand_rate, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_out * expand_rate, ch_out * expand_rate, 5, padding=2, groups=ch_out),
            bnswish(bntype)(ch_out * expand_rate, eps=bn_eps, momentum=bn_momentum),
            nn.Conv2d(ch_out * expand_rate, ch_out, 1),
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
            nn.Linear(feature // r, feature),
            nn.Sigmoid()
        ])

    def forward(self, x):
        return self.layers(x)[..., None, None] * x


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
    def __init__(self, moduletype, feature, block_features, num_cell_per_block, in_ch, out_ch, inter_feature, **kwargs):
        super(Coder, self).__init__()
        if moduletype == 'encoder':
            module = EncoderCell
            out_ch = inter_feature
        else:
            module = GenerativeCell
            in_ch = inter_feature
        layres = [nn.Conv2d(in_ch, feature * block_features[0], 1)]
        prefeature = feature
        block_features = [ch * feature for ch in block_features]
        for b_idx, feature in enumerate(block_features):
            layres.append(module(prefeature, feature, **kwargs))
            for idx in range(num_cell_per_block - 1):
                layres.append(module(feature, feature, **kwargs))
            layres.append(nn.Upsample(mode='bilinear', scale_factor=0.5 if moduletype == 'encoder' else 2))
            prefeature = feature
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
        self.moco = kwargs['moco']
        if self.moco['flag']:
            self.pre_k = None
            self.f_k_param = self.encoder.state_dict()
            self.moco_queue = nn.Parameter(
                torch.randn(kwargs['inter_feature'], self.moco['dic_size']))

    # def forward(self, x):
    #     raise ValueError('use other function')

    def load_encoder_weight(self, path):
        self.encoder.load_state_dict(torch.load(path).encoder.statedict)

    def img2latent(self, x):
        return self.encoder(x)

    def latent2img(self, z):
        return self.decoder(z)

    def img2img(self, x, grad_enc):
        with torch.set_grad_enabled(grad_enc):
            x = self.encoder(x)
        # print(x.shape)
        return self.decoder(x)

    def trainenc_dec(self, x, loss, device):
        x = x.to(device)
        output = self.img2img(x, grad_enc=True)
        return loss(output, x), output.cpu()

    def trainmoco(self, x, device):
        imgaugq, imgaugk = x
        imgaugk = imgaugk.to(device)
        imgaugq = imgaugq.to(device)
        B, _, _, _ = imgaugq.shape
        # save for queue
        if self.pre_k != None:
            self.moco_queue.data = torch.cat([self.moco_queue[:,:-self.pre_B], self.pre_k.view(self.pre_B,-1).transpose(0,1)],dim=1)
        self.f_q_param = self.encoder.cpu().state_dict()
        for f_k_p_key, f_q_p_key in zip(self.f_k_param, self.f_q_param):
            assert f_k_p_key == f_k_p_key
            self.f_k_param[f_k_p_key] = self.moco['m'] * self.f_k_param[f_k_p_key] + (1 - self.moco['m']) * \
                                        self.f_q_param[f_q_p_key]
        with torch.no_grad():
            self.encoder.load_state_dict(self.f_k_param)
            self.encoder.to(device)
            k = self.encoder(imgaugk)
            self.pre_k = k
            self.pre_B=B
        self.encoder.load_state_dict(self.f_q_param)
        self.encoder.to(device)
        q = self.encoder(imgaugq)
        B, C, _, _ = q.shape
        l_pos = torch.bmm(q.view(B, 1, C), k.view(B, C, 1)).view(-1, 1)
        l_neg = torch.mm(q.view(B, C), self.moco_queue.view(C, self.moco['dic_size']))
        logits = torch.cat([l_pos, l_neg], dim=1) / self.moco['t']
        return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long).to(device))


if __name__ == '__main__':
    # out=BatchNorm2DSwish.apply(x)
    # out.mean().backward()
    # print(bn.saved_tensors)
    import yaml
    with open('model/config/vae.yaml') as f:
        cfg=yaml.load(f)
    m = AutoEncoder(
        **cfg['vae']
    ).cuda()
    data = torch.randn(2, 3, 512, 512,device='cuda')
    output = m.img2img(data,True)
    print(output.shape)
    # from torch.utils.tensorboard import SummaryWriter
    # w=SummaryWriter()
    # w.add_graph(m,data)
    # w.close()
    # torch.onnx.export(m,data,'tmp.onnx',opset_version=11)
