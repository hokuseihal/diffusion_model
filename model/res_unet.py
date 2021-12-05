import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_ch, out_ch, kernal):
    if kernal == 1:
        return nn.Conv2d(in_ch, out_ch, kernal)
    elif kernal == 3:
        return nn.Conv2d(in_ch, out_ch, kernal, padding=1)
    else:
        assert False, "The kernel must be 3 or 1."


class DeepRes(nn.Module):
    def __init__(self, in_ch, feature, scale, kernel1):
        super(DeepRes, self).__init__()
        self.scale = scale
        self.conv1_1 = conv(in_ch, feature, 1)
        self.conv1_2 = conv(feature, in_ch, 1)
        self.conv3_1 = conv(feature, feature, 1 if kernel1 else 3)
        self.conv3_2 = conv(feature, feature, 1 if kernel1 else 3)
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.norm2 = nn.GroupNorm(1, feature)
        self.norm3 = nn.GroupNorm(1, feature)
        self.norm4 = nn.GroupNorm(1, feature)
        self.feture = feature

    def forward(self, x):
        _x = F.interpolate(x, scale_factor=self.scale)
        x = self.norm1(x)
        x = F.hardswish(x)
        x = self.conv1_1(x)
        x = self.norm2(x)
        x = F.hardswish(x)
        x = F.interpolate(x, scale_factor=self.scale)
        x = self.conv3_1(x)
        x = self.norm3(x)
        x = F.hardswish(x)
        x = self.conv3_2(x)
        x = self.norm4(x)
        x = F.hardswish(x)
        x = self.conv1_2(x)
        return x + _x


class Hopper(nn.Module):
    def __init__(self, times, in_ch, feature, scale, kernel1):
        super(Hopper, self).__init__()

        def layers():
            ret = [nn.GroupNorm(1, feature),
                   nn.Hardswish(),
                   conv(feature, feature, 3)]
            ret.insert(-1 if scale == 0.5 else -2, nn.Upsample(scale_factor=scale, mode='bilinear'))
            return ret

        self.layers = []
        for _ in range(times):
            self.layers.extend([DeepRes(in_ch=in_ch, feature=feature, scale=scale, kernel1=kernel1)])
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, embch, dropout, outch, group, isclsemb, activate, kernel1):
        super(ResBlock, self).__init__()
        outch = in_ch if outch is None else outch
        self.isclsemb = isclsemb
        self.norm1 = nn.GroupNorm(group, in_ch)
        self.conv1 = conv(in_ch, outch, 1 if kernel1 else 3)
        self.emb_proj = nn.Linear(embch, outch)
        self.norm2 = nn.GroupNorm(group, outch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv(outch, outch, 1 if kernel1 else 3)
        self.shortcut = conv(in_ch, outch, 1)
        self.activate = activate

    def forward(self, x, emb):
        _x = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        emb = self.emb_proj(self.activate(emb))[:, :, None, None]
        if self.isclsemb:
            temb, cemb = emb.chunk(2)
            x = (1 + temb) * x + cemb
        else:
            x = x + emb
        x = self.norm2(x)
        x = self.activate(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + self.shortcut(_x)


class AttnBlock(nn.Module):
    def __init__(self, feature, nhead=4):
        super(AttnBlock, self).__init__()
        self.feature = torch.tensor(feature)
        self.v = conv(feature, feature, 1)
        self.q = conv(feature, feature, 1)
        self.k = conv(feature, feature, 1)
        self.out = conv(feature, feature, 1)
        # self.attn = nn.TransformerEncoderLayer(feature, nhead, activation='gelu', dim_feedforward=dimff)
        self.norm = nn.GroupNorm(32, feature)

    def forward(self, x):
        _x = x
        x = self.norm(x)
        return _x + self.out(torch.einsum('bhwnm,bcnm->bchw', F.softmax(
            torch.einsum('bchw,bcij->bhwij', self.q(x), self.k(x)) / self.feature.float().sqrt()), self.v(x)))


class Res_AttnBlock(nn.Module):
    def __init__(self, layers):
        super(Res_AttnBlock, self).__init__()
        self.res = nn.ModuleList(layers)

    def forward(self, x, emb):
        for l in self.res:
            if l.__class__ == ResBlock:
                x = l(x, emb)
            else:
                x = l(x)
        return x


class Res_UNet(nn.Module):
    def __init__(self, in_ch, feature, embch, size, bottle_attn, activate, attn_res=(), chs=(1, 2, 4),
                 num_res_block=1,
                 dropout=0, group=32,
                 isclsemb=False, out_ch=3, hopper=False, hopper_ch=8, kernel1=False,**kwargs):
        self.kernel1 = kernel1
        super(Res_UNet, self).__init__()
        self.hopper = hopper
        if activate is None:
            activate = nn.Hardswish()
        elif activate == 'silu':
            activate = nn.SiLU()
        self.emb = nn.Sequential(
            nn.Conv1d(embch, embch, 1, groups=2 if isclsemb else 1),
            activate,
            nn.Conv1d(embch, embch, 1, groups=2 if isclsemb else 1),
        )
        _res_ch = lambda ch, outch=None: ResBlock(in_ch=ch, outch=outch, embch=embch, activate=activate, group=group,
                                                  isclsemb=isclsemb, dropout=dropout, kernel1=kernel1)
        self.convin = conv(in_ch, feature, 3)
        bottle = [
            _res_ch(feature * chs[-1]),
            _res_ch(feature * chs[-1])]
        if bottle_attn:
            bottle.insert(1, AttnBlock(feature * chs[-1]))
        self.bottolnec = Res_AttnBlock(bottle)
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        prech = 1
        res = size
        for ch in chs:
            _down = []
            _up = []
            if res in attn_res: _up.append(AttnBlock(feature * prech))
            for idx in range(num_res_block)[::-1]:
                _down.append(_res_ch(feature * prech, feature * ch))
                _up.insert(0, _res_ch(feature * ch * (2 if idx == 0 else 1), feature * prech))
                prech = ch
            if res in attn_res: _down.append(AttnBlock(feature * ch))
            self.down.append(Res_AttnBlock(_down))
            self.up.insert(0, Res_AttnBlock(_up))
            res //= 2
        self.hopper_down = Hopper(times=3, in_ch=3, feature=hopper_ch,
                                  scale=0.5, kernel1=kernel1) if hopper else nn.Sequential()
        self.hopper_up = Hopper(times=3, in_ch=3, feature=hopper_ch, scale=2,
                                kernel1=kernel1) if hopper else nn.Sequential()
        self.out = nn.Sequential(
            nn.GroupNorm(group, feature * chs[0]),
            activate,
            conv(feature * chs[0], out_ch, 1)
        )

    def forward(self, x, emb, get_latent=False):
        skips = []

        emb = self.emb(emb[:, :, None])[:, :, 0]
        x = self.hopper_down(x)
        x = self.convin(x)
        for idx, l in enumerate(self.down):
            x = l(x, emb)
            skips.append(x)
            if idx < len(self.down) - 1 and not self.kernel1: x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        if get_latent: return skips
        for idx, l in enumerate(self.up):
            tmp = skips.pop(-1)
            x = l(torch.cat([x, tmp], dim=1), emb)
            if idx < len(self.up) - 1 and not self.kernel1: x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.out(x)
        x = self.hopper_up(x)
        return x


if __name__ == '__main__':
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                profile_memory=True) as p:
        size = 128
        batchsize = 16
        m = Res_UNet(in_ch=3, feature=128, size=size, embch=64, chs=(1, 1, 1, 2, 2, 2), hopper=False,
                     attn_res=(32, 16, 8), activate='silu', bottle_attn=False, kernel1=False).cuda()
        print(m)
        x = torch.randn(batchsize, 3, size, size).cuda()
        temb = torch.randn(batchsize, 64).cuda()
        output = m(x, temb)
        print(output.dtype)
    print(output.shape)
