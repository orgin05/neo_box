import mne
from mne.time_frequency import psd_array_multitaper
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .utils import cut_epochs
from .feature_extractors import filterbank_extractor


def plot_embeddings(embd, y, event_id, size=20, figsize=(4, 5), show_legend=True):
    fig, ax = plt.subplots(figsize=figsize)
    for label in event_id.keys():
        l = event_id[label]
        idx = y == l
        ax.scatter(embd[idx, 0], embd[idx, 1], s=size, label=label)

    ax.set_xlabel(r"PC1")
    ax.set_ylabel(r"PC2")
    if show_legend:
        ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig


def snapshot_brain(fig_3d, info, data=None, show_name=False):
    if data is not None:
        cmap = mpl.cm.viridis
        norm = mpl.colors.Normalize(vmin=data.min(), vmax=data.max())
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    directions = [0, 180]  # right, left
    figs = []
    for d in directions:
        # right
        mne.viz.set_3d_view(fig_3d, azimuth=d, elevation=70)
        xy, im = mne.viz.snapshot_brain_montage(fig_3d, info, hide_sensors=False)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(im, interpolation='none')
        ax.set_axis_off()
        if data is not None:
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(mappable, cax=cbar_ax)
        if show_name:
            xy_pts = np.vstack([xy[ch] for ch in info["ch_names"]])
            for i, pos in enumerate(xy_pts):
                ax.text(*pos, i, color='white')

        figs.append(fig)

    return figs


def plot_time_frequency(data, events, sfreq, freqs, epoch_time_range, event_id):
    """
    data: numpy.ndarray, (n_ch, n_times)
    events: ndarray (n_events, 3), the first column is onset index, the second is duration, and the third is event type
    freqs: numpy.ndarray, frequency bands to filter
    epoch_time_range: tuple, (t_onset, t_offset)
    event_id: dict {id: name}
    """
    # extract power, (n_ch, n_freqs, n_times)
    power = filterbank_extractor(data, sfreq, freqs, reshape_freqs_dim=False)
    power = 10 * np.log10(power)
    # normalize by freqs
    power -= power.mean(axis=(0, 2), keepdims=True)
    power /= power.std(axis=(0, 2), keepdims=True)
    # image vlim
    mean_, std_ = power.mean(), power.std()
    # cut epochs
    epochs = cut_epochs((*epoch_time_range, sfreq), power, events[:, 0])
    # average by event type
    classes = np.unique(events[:, -1])
    fig, axes = plt.subplots(1, len(classes), figsize=(10, 5))
    for ax, y_ in zip(axes, classes):
        average_power = epochs[events[:, -1] == y_].mean(axis=(0, 1)) # keep freqencies and times
        im = ax.imshow(average_power, cmap='RdBu_r', 
                        vmin=mean_ - 0.5 * std_, 
                        vmax=mean_ + 0.5 * std_,
                        aspect='auto',
                        origin='lower')
        ax.set_xticks(np.linspace(-0.5, average_power.shape[1] - 0.5, 5))
        ax.set_xticklabels([f'{i:.2f}' for i in np.linspace(*epoch_time_range, 5)])
        ax.set_yticks(np.linspace(-0.5, average_power.shape[0] - 0.5, 10))
        ax.set_yticklabels([f'{int(i):3d}' for i in np.linspace(freqs[0], freqs[-1], 10)])
        ax.set_title(event_id[y_])
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig


def plot_ersd(data, events, sfreq, epoch_time_range, event_id, rest_event=0):
    n_ch = data.shape[0]
    event_desc = {v:k for k, v in event_id.items()}
    epochs = cut_epochs((*epoch_time_range, sfreq), data, events[:, 0])

    psd, freqs = psd_array_multitaper(epochs, sfreq, fmin=0, fmax=200, bandwidth=15)

    mean_psd_rest = psd[events[:, -1] == rest_event].mean(axis=0)
    ersds = []
    for e in np.unique(events[:, -1]):
        if e != rest_event:
            mean_psd = psd[events[:, -1] == e].mean(axis=0)
            ersd = mean_psd / mean_psd_rest - 1
            ersds.append((event_desc[e], ersd))
    fig, axes = plt.subplots(n_ch, len(ersds), figsize=(3 * len(ersds), n_ch), sharex=True, sharey=True)

    for i in range(n_ch):
        if len(ersds) == 1:
            for j, ersd in enumerate(ersds):
                if i == 0:
                    axes[i].set_title(ersd[0])
                axes[i].plot(freqs, ersd[1][i])
                axes[i].set_ylabel(f'ch_{i + 1}')
                axes[i].axhline(0, color='gray', linestyle='--')
        else:
            for j, ersd in enumerate(ersds):
                if i == 0:
                    axes[i, j].set_title(ersd[0])
                axes[i, j].plot(freqs, ersd[1][i])
                axes[i, j].set_ylabel(f'ch_{i + 1}')
                axes[i, j].axhline(0, color='gray', linestyle='--')
    fig.suptitle('ERSD')
    return fig


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    return disp.figure_


def plot_states(time_range, pred_states, ax, colors=None):
    classes = np.unique(pred_states)
    if colors is None:
        colors = [plt.get_cmap('tab10')(i)[:3] for i in range(len(classes))]
    for i, c in enumerate(classes):
        ax.fill_between(np.linspace(*time_range, len(pred_states)), 0, 1,
                        where=(pred_states == c), alpha=0.6, color=colors[i])
    return ax


def plot_state_prob_with_cue(time_range, true_states, pred_probs, ax, colors=None):
    # normalize
    ax.plot(np.linspace(*time_range, len(pred_probs)), pred_probs, color='k')
    # for each class, fill different colors
    classes = np.unique(true_states)
    if colors is None:
        colors = [plt.get_cmap('tab10')(i)[:3] for i in range(len(classes))]
    for i, c in enumerate(classes):
        ax.fill_between(np.linspace(*time_range, len(true_states)), 0, 1,
                        where=(true_states == c), alpha=0.6, color=colors[i])
    return ax
