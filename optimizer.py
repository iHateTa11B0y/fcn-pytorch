from torch import optim

class OptimizerScheduler(object):
    def __init__(self, model, cfg):
        super(OptimizerScheduler, self).__init__()
        self.optimizer = optim.SGD(model.parameters(), 
                                  lr=cfg.SOLVER.LR, 
                                  momentum=cfg.SOLVER.MOMENTUM,
                                  weight_decay=cfg.SOLVER.WEIGHT_DECAY)
        self.lr = cfg.SOLVER.LR
        self.steps = cfg.SOLVER.STEPS

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, iteration):
        if iteration in self.steps:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.1
                self.lr *= 0.1
        self.optimizer.step()

