import math

import numpy as np
import torch


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
    def __init__(self, denoizer, criterion, optimizer, schedule, device, lr, embch=32, type='ddpm', n_iter=1000,
                 beta=(1e-4, 2e-2)):
        self.device = device
        self.denoizer = denoizer
        self.criterion = criterion
        self.type = type
        self.embch = embch
        self.optimizer = optimizer(self.denoizer.parameters(), lr=lr)
        self.n_iter = n_iter
        if self.type == 'ddpm':
            self.nextsample = self.ddpmnextsample
        if schedule == 'cos':
            f = lambda t, s=1e-3: torch.cos((t / self.n_iter + s) / (1 + s) * np.pi / 2)
            self.a = (f(torch.linspace(*beta, self.n_iter)) / f(0)).to(self.device)
            self.b = (torch.clip(1 - self.a / ([1] + self.a[:-1]), 0.999)).to(self.device)
        elif schedule == 'linear':
            self.b = torch.linspace(*beta, self.n_iter).to(self.device)
            self.a = torch.cumprod(1 - self.b, -1).to(self.device)
            self._a = 1 - self.b

    def trainbatch(self, x):
        B, C, H, W = x.shape
        T = torch.randint(self.n_iter, (B,))
        t = get_timestep_embedding(T, self.embch).to(self.device)
        e = torch.rand_like(x).to(self.device)
        xt = self.a[T].view(-1, 1, 1, 1).sqrt() * x + (1 - self.a[T].view(-1, 1, 1, 1)).sqrt() * e
        output = self.denoizer(xt, t)
        loss = self.criterion(e, output)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {'loss': loss.item()}

    @torch.no_grad()
    def sample(self, stride, embch, shape=None, x=None):
        assert not (shape is None and x is None)
        if x is None: x = torch.rand(shape).to(self.device)
        for t in torch.range(self.n_iter-1, 0, -stride, dtype=torch.long):
            print(f'\rsampling:{t}', end='')
            ys = get_timestep_embedding(t.view(1), embch).to(self.device)
            et = self.denoizer(x, ys)
            x = self.nextsample(x, et, t)
        return x

    def ddpmnextsample(self, x, et, t):
        c = (1 - self.a[t - 1]) / (1 - self.a[t]) * (1 - self._a[t])
        return 1 / self._a[t].sqrt() * (x - (1 - self._a[t]) / (1 - self.a[t].sqrt()) * et) + c * torch.rand_like(x)
