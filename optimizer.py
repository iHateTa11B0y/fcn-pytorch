from torch import optim

class OptimizerScheduler(object):
    def __init__(self, model, base_lr, steps):
        super(OptimizerScheduler, self).__init__()
        self.optimizer = optim.RMSprop(model.parameters(), lr=base_lr, weight_decay=1e-8)
        self.lr = base_lr
        self.steps = steps

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, iteration):
        if iteration in self.steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
                self.lr *= 0.1
        self.optimizer.step()

