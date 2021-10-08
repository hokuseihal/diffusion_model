from itertools import product

import torch
import torch.nn.functional as F


def take_err(x, idx):
    s0, s1, s2, s3 = x.shape
    idx = idx.view(-1)
    return x[torch.arange(s0).repeat_interleave(s1), torch.arange(s1).repeat(s0), idx].reshape(s0, s1, s3)


def motion_estimate(img0, img1, kernel_size, search, p, up):
    B, C, H, W = img0.shape
    k = kernel_size
    assert img0.shape[2] % k == 0
    assert img0.shape[3] % k == 0
    assert img1.shape[2] % k == 0
    assert img1.shape[3] % k == 0
    if search == 'full':
        # print(img1)
        p *= up
        H *= up
        W *= up
        k *= up
        # dst = F.interpolate(img1, scale_factor=up, mode='bicubic')
        dst = img1
        _dst=img1.clone()
        dst = F.pad(dst, (p, p, p, p), value=256)
        dst = F.unfold(dst, k + p * 2, stride=k)
        dst = dst.reshape(B, C, k + p * 2, k + p * 2, H // k, W // k)
        dst = dst.permute(0, 4, 5, 1, 2, 3)
        dst = dst.reshape(B * (H // k) * (W // k), C, k + p * 2, k + p * 2)
        dst = F.unfold(dst, k // up)
        dst = dst.reshape(B, (H // k) * (W // k), C * (k // up) ** 2, (p * 2 + (k - k // up) + 1) ** 2)
        dst = dst.permute(0, 1, 3, 2)
        # print(dst)
        p //= up
        H //= up
        W //= up
        k //= up

        src = img0
        src = F.unfold(src, k, stride=k).permute(0, 2, 1).unsqueeze(2)
        # print(src.shape)
        # print(dst.shape)

        err = (src - dst).abs()
        idx = err.mean(dim=-1).argmin(dim=-1)
        nump = 2 * p + 1
        x = (idx // nump - p) / p
        y = (idx % nump - p) / p
        x = x.reshape(B, 1, H // k, H // k)
        y = y.reshape(B, 1, W // k, W // k)
        motion = torch.cat([x, y], dim=1)
        src = src.reshape(B, (H // k) * (W // k), C, k, k)
        est_dst = torch.zeros(B,C,H,W).to(src.device)
        for b in range(B):
            for i, j in product(range(H // k), range(W // k)):
                mx = motion[b, 0, i, j]
                my = motion[b, 1, i, j]
                x0 = (i * k + mx * p).long()
                y0 = (j * k + my * p).long()
                x1 = x0 + p
                y1 = y0 + p
                est_dst[b, :, x0:x1, y0:y1] = src[b, i * 8 + j]
        img_err = _dst - est_dst
        # print(img_err)

        return img_err, motion


if __name__ == '__main__':
    import time

    start = time.time()
    for i in range(80):
        # x = torch.arange(4*3*64*64).float().view(4,3,64,64).cuda()
        # print(i)
        x = torch.randn(4, 3, 64, 64).cuda()
        # y = torch.randn(4, 3, 128, 128)
        motion_estimate(x, x, kernel_size=16, p=16, up=1, search='full')
    print(time.time() - start)
