import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, feature, nhead=4):
        super(AttnBlock, self).__init__()
        self.feature = torch.tensor(feature)
        self.v = nn.Conv2d(feature, feature, 1)
        self.q = nn.Conv2d(feature, feature, 1)
        self.k = nn.Conv2d(feature, feature, 1)
        self.out = nn.Conv2d(feature, feature, 1)
        # self.attn = nn.TransformerEncoderLayer(feature, nhead, activation='gelu', dim_feedforward=dimff)
        self.norm=nn.GroupNorm(32,feature)

    def forward(self, x):
        _x=x
        x=self.norm(x)
        return _x+self.out(torch.einsum('bhwnm,bcnm->bchw', F.softmax(
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
    def __init__(self, in_ch, feature, embch, size, activate=nn.Hardswish(), attn_res=(), chs=(1, 2, 4),
                 num_res_block=1,
                 dropout=0, group=32,
                 isclsemb=False, out_ch=3):
        super(Res_UNet, self).__init__()
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
        self.out = nn.Sequential(
            nn.GroupNorm(group, feature * chs[0]),
            activate,
            nn.Conv2d(feature * chs[0], out_ch, 3, 1, 1)
        )

    def forward(self, x, emb):
        skips = []

        emb = self.emb(emb[:, :, None])[:, :, 0]
        x = self.convin(x)
        for idx, l in enumerate(self.down):
            x = l(x, emb)
            skips.append(x)
            if idx < len(self.down) - 1: x = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        for idx, l in enumerate(self.up):
            tmp = skips.pop(-1)
            x = l(torch.cat([x, tmp], dim=1), emb)
            if idx < len(self.up) - 1: x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.out(x)
        return x


if __name__ == '__main__':
    device = 'cuda'
    m = Res_UNet(in_ch=3, feature=128, size=128, embch=64).to(device)
    print(m)
    with torch.cuda.amp.autocast():
        x = torch.randn(8, 3, 64, 64).to(device)
        temb = torch.randn(8, 64).to(device)
        output = m(x, temb)
        print(output.dtype)
    print(output.shape)