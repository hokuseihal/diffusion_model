class EMA_scheduler:
    def __init__(self, optimizer, p=0.5, numstep=50, mu_factor=0.75, factor=0.5, min_lr=0, verbose=False,maxloss=100):
        self.optimizer = optimizer
        self.verbose = verbose
        self.min_lr = min_lr
        self.factor = factor
        self.p = p
        self.numstep = numstep
        self.mu_factor = mu_factor
        self.mu = maxloss
        self.high_low = [0, 0]
        self.last_max_mu = maxloss

    def step(self, loss):
        _mu = self.mu * self.p + loss * (1 - self.p)
        self.mu=_mu
        if (self.last_max_mu * self.mu_factor > self.mu):
            if(self.verbose):
                print(f'mu changed:{self.last_max_mu:.3f}->{self.mu:.3f}')
            self.high_low = [0, 0]
            self.last_max_mu=self.mu
        self.high_low[loss > self.mu] += 1
        if (min(self.high_low) > self.numstep):
            self.high_low = [0, 0]
            self.last_max_mu = self.mu
            for i, param_group in enumerate(self.optimizer.param_groups):
                old_lr = float(param_group['lr'])
                new_lr = max(old_lr * self.factor, self.min_lr)
                param_group['lr'] = new_lr
                if self.verbose:
                    print('reducing learning rate of group {} to {:.4e}.'.format(i, new_lr))
