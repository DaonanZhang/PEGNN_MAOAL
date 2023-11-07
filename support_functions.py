import os
import mpl_scatter_density
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import random
import numpy as np
import torch


def make_dir(path):
    try: 
        os.mkdir(path)
    except: 
        pass
    
    
def build_folder_and_clean(path):
    check = os.path.exists(path)
    if check:
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(path)
        
        
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def save_square_img(contents, xlabel, ylabel, savename, title):
    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ], N=256)
    
    plt.clf()
    plt.rcParams['font.size'] = 15
    
    max_value = max(contents[0].max(), contents[1].max())
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')
    fig.set_size_inches(7, 6)
    ax.set_position([0, 0, 0.8, 1])
    
    density = ax.scatter_density(contents[0], contents[1], cmap=white_viridis)
    fig.colorbar(density, label='Number of points')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    
    ax.set_xlim([0, max_value])
    ax.set_ylim([0, max_value])
    ax.plot([0, max_value], [0, max_value], color='k')
    fig.savefig("%s.png" %(savename))
    plt.close(fig)
    
    
def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu().numpy())
            max_grads.append(p.grad.abs().max().detach().cpu().numpy())
    plt.figure(figsize=(10, 6))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.4, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.tick_params(axis='x', labelsize=8)    # 设置x轴标签大小
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()
    