import numpy as np


def plot_masked_tfr(T, m, ax, labels={}, kwargs={}):
    ax.imshow(np.repeat(T, 1, -1), origin='lower', cmap='Greys')
    sig_T = np.copy(T)
    sig_T[np.logical_not(m)]=np.nan
    im = ax.imshow(np.repeat(sig_T, 1, -1), origin='lower', cmap='Spectral_r', **kwargs);
    if 'x' in labels.keys():
        xticks,xlabels = labels.get('x')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
    if 'y' in labels.keys():
        yticks,ylabels = labels.get('y')
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels)
    return im
