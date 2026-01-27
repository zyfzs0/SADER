import numpy as np
import torch


def adjust_learning_rate(optimizer, epoch, args):
    """
    Adjusts learning rate according to pre-defined schedule.
    Suitable for simple models like CNN or MAE during pretraining.
    
    Args:
        optimizer: torch optimizer
        epoch: current epoch number (1-based)
        args: argument object with attributes:
            - learning_rate: base LR
            - lradj: type of LR schedule ('exp', 'step', 'cosine')
            - lr_decay: decay factor for exponential
            - lr_step: list of epochs to step down
            - min_lr: minimum learning rate
    """
    base_lr = args.learning_rate
    new_lr = base_lr

    if args.lradj == 'exp':
        # exponential decay: LR = base_lr * decay^(epoch-1)
        decay = getattr(args, 'lr_decay', 0.95)
        new_lr = base_lr * (decay ** (epoch - 1))

    elif args.lradj == 'step':
        # step decay: reduce LR at specific epochs
        lr_step = getattr(args, 'lr_step', [10, 20, 30])
        lr_factor = getattr(args, 'lr_factor', 0.5)
        for step_epoch in lr_step:
            if epoch >= step_epoch:
                new_lr *= lr_factor

    elif args.lradj == 'cosine':
        # cosine annealing
        total_epochs = getattr(args, 'train_epochs', 50)
        new_lr = base_lr * 0.5 * (1 + np.cos(np.pi * (epoch - 1) / total_epochs))

    # clip to min_lr if defined
    min_lr = getattr(args, 'min_lr', 1e-6)
    new_lr = max(new_lr, min_lr)

    # update optimizer
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

    print(f"[LR Scheduler] Epoch {epoch}: setting learning rate to {new_lr:.6f}")

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss