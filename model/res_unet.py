import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepRes(nn.Module):
    def __init__(self, in_ch, feature, scale):
        super(DeepRes, self).__init__()
        self.scale = scale
        self.conv1_1 = nn.Conv2d(in_ch, feature, 1)
        self.conv1_2 = nn.Conv2d(feature, in_ch, 1)
        self.conv3_1 = nn.Conv2d(feature, feature, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(feature, feature, 3, 1, 1)
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
    def __init__(self, times, in_ch, feature, scale):
        super(Hopper, self).__init__()

        def layers():
            ret = [nn.GroupNorm(1, feature),
                   nn.Hardswish(),
                   nn.Conv2d(feature, feature, 3, 1, 1)]
            ret.insert(-1 if scale == 0.5 else -2, nn.Upsample(scale_factor=scale, mode='bilinear'))
            return ret

        self.layers = []
        for _ in range(times):
            self.layers.extend([DeepRes(in_ch=in_ch, feature=feature, scale=scale)])
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, embch, dropout, outch, group, isclsemb, activate):
        super(ResBlock, self).__init__()
        outch = in_ch if outch is None else outch
        self.isclsemb = isclsemb
        self.norm1 = nn.GroupNorm(group, in_ch)
        self.conv1 = nn.Conv2d(in_ch, outch, 3, 1, 1)
        self.emb_proj = nn.Linear(embch, outch)
        self.norm2 = nn.GroupNorm(group, outch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, 1)
        self.shortcut = nn.Conv2d(in_ch, outch, 1, 1, 0)
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
    def __init__(self, feature, dimff=None, nhead=4, resize=8):
        super(AttnBlock, self).__init__()
        if dimff is None: dimff = feature
        self.resize = resize
        self.attn = nn.TransformerEncoderLayer(feature, nhead, activation='gelu', dim_feedforward=dimff)

    def forward(self, x):
        _x = x
        shape = x.shape
        if self.resize: x = F.interpolate(x, self.resize)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(2, 0, 1)
        x = self.attn(x)
        x = x.permute(1, 2, 0).view(B, C, H, W)
        if self.resize: x = F.interpolate(x, shape[-2:])
        return _x + x


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
    def __init__(self, in_ch, feature, embch, size, activate=nn.Hardswish(), attn_res=(), chs=(1, 2, 4),
                 num_res_block=1,
                 dropout=0, group=32,
                 isclsemb=False, out_ch=3, hopper=False,hopper_ch=8):
        super(Res_UNet, self).__init__()
        self.hopper = hopper
        self.emb = nn.Sequential(
            nn.Conv1d(embch, embch, 1, groups=2 if isclsemb else 1),
            activate,
            nn.Conv1d(embch, embch, 1, groups=2 if isclsemb else 1),
        )
        _res_ch = lambda ch, outch=None: ResBlock(in_ch=ch, outch=outch, embch=embch, activate=activate, group=group,
                                                  isclsemb=isclsemb, dropout=dropout)
        self.convin = nn.Conv2d(in_ch, feature, 3, 1, 1)
        self.bottolnec = Res_AttnBlock([
            _res_ch(feature * chs[-1]),
            AttnBlock(feature * chs[-1]),
            _res_ch(feature * chs[-1])
        ])
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
                                  scale=0.5) if hopper else nn.Sequential()
        self.hopper_up = Hopper(times=3, in_ch=3, feature=hopper_ch, scale=2) if hopper else nn.Sequential()
        self.out = nn.Sequential(
            nn.GroupNorm(group, feature * chs[0]),
            activate,
            nn.Conv2d(feature * chs[0], out_ch, 3, 1, 1)
        )

    def forward(self, x, emb):
        skips = []

        emb = self.emb(emb[:, :, None])[:, :, 0]
        x = self.hopper_down(x)
        x = self.convin(x)
        for idx, l in enumerate(self.down):
            x = l(x, emb)
            skips.append(x)
            if idx < len(self.down) - 1: x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x = self.bottolnec(x, emb)
        for idx, l in enumerate(self.up):
            tmp = skips.pop(-1)
            x = l(torch.cat([x, tmp], dim=1), emb)
            if idx < len(self.up) - 1: x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.out(x)
        x = self.hopper_up(x)
        return x


if __name__ == '__main__':
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                profile_memory=True) as p:
        size = 512
        batchsize=8
        m = Res_UNet(in_ch=3, feature=128, size=size, embch=64, chs=(1, 1, 1, 2), hopper=True,attn_res=(32,16,8)).cuda()
        print(m)
        x = torch.randn(batchsize, 3, size, size).cuda()
        temb = torch.randn(batchsize, 64).cuda()
        output = m(x, temb)
        print(output.shape)
    print(p.key_averages())
