from torch.optim import Optimizer

# This file contains the StepLRBase learning rate scheduler used is some of the experiments

# This is the super class for LRSscheduler from pytorch
class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# The super class has the method step, which acquires the learning rate and
# then updates is to the new learning rate provided by the subclass

class StepLRBase(_LRScheduler):
    """Sets the learning rate of each parameter group to a floor (or base) learning rate +
    the initial lr decayed by gamma every epoch. Thus the learning rate will decay exponentially
    towards the floor learning rate.

    This class is an extension of the pytorch StepLR class.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, step_size, floor_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.floor_lr = floor_lr
        super(StepLRBase, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.floor_lr + base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

class ExpDecayLRBase(_LRScheduler):
    """Sets the learning rate of each parameter group to a floor (or base) learning rate +
    the initial lr decayed by gamma every epoch. Thus the learning rate will decay exponentially
    towards the floor learning rate.

    This class is an extension of the pytorch StepLR class.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, step_size, floor_lr, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        self.floor_lr = floor_lr
        super(StepLRBase, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.floor_lr + base_lr * self.gamma ** (self.last_epoch / self.step_size)
                for base_lr in self.base_lrs]
