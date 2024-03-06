from math import cos, pi
import matplotlib.pyplot as plt
import os

def warmup_cos_iter(optimizer, epoch_iter, current_iter, max_iter,lr_min=0, lr_max=0.1, warmup=True):
    warmup_iter = 1500 if warmup else 0
    if current_iter < warmup_iter:
        lr = lr_max * current_iter / warmup_iter
    else:
        lr = lr_min + (lr_max-lr_min)*(1 -(current_iter - warmup_iter) / (max_iter - warmup_iter))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def plot_loss(x, y, x2, y2, x3, y3, title, save_dir,xlim0=None, xlim1=None, ylim0=None, ylim1=None, xlabel='lesion size', ylabel='Dice loss', figsize=(60,18)):
        plt.figure(figsize=figsize)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        if xlim0 is not None:
            plt.xlim(xlim0, xlim1)
        if ylim0 is not None:
            plt.ylim(ylim0, ylim1)
        plt.title(title, size = 70)

        plt.xlabel(xlabel, size = 50)
        plt.ylabel(ylabel, size = 50)
        plt.plot(x, y, 'ro', label='AP')
        plt.plot(x2, y2, 'o', label='PA')
        plt.plot(x3, y3, 'yo', label='AP_NTUH')
        plt.legend(fontsize=30)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(os.path.join(save_dir, f'{title}.png'))
        plt.close()

def grad_norm(model, writer, step):
    total_norm = 0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    writer.add_scalar(f'Gradient_norm', total_norm, step)
    return total_norm