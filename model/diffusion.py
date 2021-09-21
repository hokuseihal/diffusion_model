import numpy as np
import torch


class Diffusion():
    def __init__(self, denoizer, criterion, optimizer, schedule, device, type='ddpm', n_iter=1000, beta=(1e-4, 2e-2)):
        self.device = device
        self.denoizer = denoizer
        self.criterion = criterion
        self.type = type
        self.optimizer = optimizer(self.denoizer.parameters(), lr=2e-4)
        self.n_iter = n_iter
        if self.type == 'ddpm':
            self.nextsample = self.ddpmnextsample
            self.maketarget = self.maketargetddpm
        if schedule == 'cos':
            f = lambda t, s=1e-3: torch.cos((t / self.n_iter + s) / (1 + s) * np.pi / 2)
            self.a = (f(torch.linspace(*beta, self.n_iter)) / f(0)).to(self.device)
            self.b = (torch.clip(1 - self.a / ([1] + self.a[:-1]), 0.999)).to(self.device)
        elif schedule == 'linear':
            self.b = torch.linspace(*beta, self.n_iter).to(self.device)
            self.a = torch.cumprod(1 - self.b, -1).to(self.device)
            self._a = 1 - self.b

    def maketargetddpm(self, x, t):
        e = torch.rand_like(x).to(self.device)
        xt = self.a[t].view(-1, 1, 1, 1).sqrt() * x + (1 - self.a[t].view(-1, 1, 1, 1)).sqrt() * e
        return xt, e

    def trainbatch(self, x):
        B, C, H, W = x.shape
        T = torch.randint(self.n_iter, (B,))
        t = self.b[T]
        xt, e = self.maketarget(x, T)
        output = self.denoizer(xt, t.view(-1, 1))
        loss = self.criterion(e, output)
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    @torch.no_grad()
    def sample(self, n_sample_iter, shape=None, x=None):
        assert not (shape is None and x is None)
        if x is None: x = torch.rand(shape).to(self.device)
        for i, t in zip(range(n_sample_iter), range(0, self.n_iter, self.n_iter // n_sample_iter)):
            ys = self.b[t].view(1, 1)
            et = self.denoizer(x, ys)
            x = self.nextsample(x, et, t)
        return x

    def ddpmnextsample(self, x, et, t):
        c = (1 - self.a[t - 1]) / (1 - self.a[t]) * (1 - self._a[t])
        return 1 / self._a[t].sqrt() * (x - (1 - self._a[t]) / (1 - self.a[t].sqrt()) * et) + c * torch.rand_like(x)
