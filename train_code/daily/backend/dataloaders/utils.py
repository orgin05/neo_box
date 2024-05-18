import numpy as np
import mne


def upsample_events(events, upsample_interval=500):
    # Upsample events every 500 sample points
    events_new = []
    for e_ in events:
        for i in range(0, e_[1] - upsample_interval + 1, upsample_interval):
            events_new.append([e_[0] + i, 0, e_[-1]])
    return np.array(events_new)


def extend_signal(raw, frequencies, freq_band):
    """ Extend a signal with filter bank using MNE """
    raw_ext = np.vstack([
        bandpass_filter(raw, l_freq=f - freq_band, h_freq=f + freq_band)
        for f in frequencies]
    )

    info = mne.create_info(
        ch_names=sum(
            list(map(lambda f: [ch + '-' + str(f) + 'Hz'
                                for ch in raw.ch_names],
                     frequencies)), []),
        ch_types=['ecog'] * len(raw.ch_names) * len(frequencies),
        sfreq=int(raw.info['sfreq'])
    )

    return mne.io.RawArray(raw_ext, info)


def bandpass_filter(raw, l_freq, h_freq, method="iir", verbose=False):
    """ Band-pass filter a signal using MNE """
    return raw.copy().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method=method,
        verbose=verbose
    ).get_data()