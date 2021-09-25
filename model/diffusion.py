import math
from functools import partial

import numpy as np
import torch
import torch.optim.lr_scheduler as sche

import utils.ema_scheduler  as ema


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Diffusion():
    def __init__(self, denoizer, criterion, schedule, device, lr, eta, g_clip, embch=32, n_iter=1000,
                 beta=(1e-4, 2e-2)):
        self.device = device
        self.denoizer = denoizer
        self.criterion = criterion
        self.type = type
        self.embch = embch
        self.optimizer = torch.optim.Adam(self.denoizer.parameters(), lr=lr)
        # self.scheduler = ema.EMA_scheduler(self.optimizer, verbose=True)
        self.n_iter = n_iter
        self.nextsample = partial(self.ddimnextsample, eta=eta)
        # self.nextsample=self.testnextsample
        # TODO need debug
        if schedule == 'cos':
            f = lambda t, s=1e-3: np.cos((t / self.n_iter + s) / (1 + s) * np.pi / 2)
            self.a = (f(torch.linspace(*beta, self.n_iter)) / f(0)).to(self.device)
            self.b = (
                torch.clip(1 - self.a / torch.cat([torch.ones(1).to(self.device), self.a[:-1]], dim=-1), 0.999)).to(
                self.device)
        elif schedule == 'linear':
            self.b = torch.linspace(*beta, self.n_iter).to(self.device)
            self.a = torch.cumprod(1 - self.b, -1).to(self.device)
            self._a = 1 - self.b
        self.g_clip = g_clip

    def trainbatch(self, x):
        B, C, H, W = x.shape
        T = torch.randint(self.n_iter, (B,))
        t = get_timestep_embedding(T, self.embch).to(self.device)
        e = torch.rand_like(x).to(self.device)
        xt = self.a[T].view(-1, 1, 1, 1).sqrt() * x + (1 - self.a[T].view(-1, 1, 1, 1)).sqrt() * e
        target=e
        # target=(1 - self.a[T].view(-1, 1, 1, 1)).sqrt() * e
        output = self.denoizer(xt, t)
        loss = self.criterion(target, output)
        loss.backward()
        torch.nn.utils.clip_grad_norm(
            self.denoizer.parameters(), self.g_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        # self.scheduler.step(loss)
        return {'loss': loss.item()}

    @torch.no_grad()
    def sample(self, stride, embch, shape=None, x=None):
        assert not (shape is None and x is None)
        if x is None: x = torch.rand(shape).to(self.device)
        for t in torch.arange(self.n_iter - 1, 1, -stride, dtype=torch.long):
            print(f'\rsampling:{t}', end='')
            ys = get_timestep_embedding(t.view(1), embch).to(self.device)
            et = self.denoizer(x, ys)
            x = self.nextsample(x, et, t)
        return x

    def ddpmnextsample(self, x, et, t):
        assert t >= 1
        c = (1 - self.a[t - 1]) / (1 - self.a[t]) * (1 - self._a[t])
        return 1 / self._a[t].sqrt() * (x - (1 - self._a[t]) / (1 - self.a[t].sqrt()) * et) + c * torch.rand_like(x)

    def ddimnextsample(self, xt, et, t, eta):
        assert t >= 1
        c1 = eta * ((1 - self.a[t] / self.a[t - 1]) * (1 - self.a[t - 1]) / (1 - self.a[t])).sqrt()
        c2 = ((1 - self.a[t - 1]) - c1 ** 2).sqrt()
        x0_t = (xt - (1 - self.a[t]).sqrt() * et) / self.a[t].sqrt()
        return self.a[t - 1].sqrt() * x0_t + c1 * torch.randn_like(xt) + c2 * et

    def testnextsample(self, xt, et, t):
        assert t >= 1
        return (xt - et) / self.a[t]