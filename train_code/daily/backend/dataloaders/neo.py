import numpy as np
import os
import json
import mne
import glob
import pyedflib
from .utils import upsample_events
from settings.config import settings

FINGERMODEL_IDS = settings.FINGERMODEL_IDS
FINGERMODEL_IDS_INVERSE = settings.FINGERMODEL_IDS_INVERSE

CONFIG_INFO = settings.CONFIG_INFO


def raw_loader(data_root, session_paths:dict, 
                      reref_method='monopolar',
                      use_ori_events=False,
                      upsampled_epoch_length=1., 
                      ori_epoch_length=5):
    """
    Params:
        data_root: 
        session_paths: dict of lists
        reref_method (str): rereference method: monopolar, average, or bipolar
        upsampled_epoch_length (None or float): None: do not do upsampling
        ori_epoch_length (int, dict, or 'varied'): original epoch length in second
    """
    raws_loaded = load_sessions(data_root, session_paths, reref_method)
    # process event
    raws = []
    event_id = {}
    for (finger_model, raw) in raws_loaded:
        fs = raw.info['sfreq']
        {d: int(d) for d in np.unique(raw.annotations.description)}
        events, _ = mne.events_from_annotations(raw, event_id={d: int(d) for d in np.unique(raw.annotations.description)})

        event_id = event_id | {FINGERMODEL_IDS_INVERSE[int(d)]: int(d) for d in np.unique(raw.annotations.description)}
        
        if isinstance(ori_epoch_length, int) or isinstance(ori_epoch_length, float):
            trial_duration = ori_epoch_length
        elif ori_epoch_length == 'varied':
            trial_duration = None
        elif isinstance(ori_epoch_length, dict):
            trial_duration = ori_epoch_length
        else:
            raise ValueError(f'Unsupported epoch_length {ori_epoch_length}')
        
        events = reconstruct_events(events, fs, 
                                    use_ori_events=use_ori_events,
                                    trial_duration=trial_duration)
        if upsampled_epoch_length is not None:
            events = upsample_events(events, int(fs * upsampled_epoch_length))
        
        event_desc = {e: FINGERMODEL_IDS_INVERSE[e] for e in np.unique(events[:, 2])}
        annotations = mne.annotations_from_events(events, fs, event_desc)
        raw.set_annotations(annotations)
        raws.append(raw)

    raws = mne.concatenate_raws(raws)

    raws.load_data()

    return raws, event_id


def reref(raw, method='average'):
    if method == 'average':
        return raw.set_eeg_reference('average')
    elif method == 'bipolar':
        anode = CONFIG_INFO['strips'][0] + CONFIG_INFO['strips'][1][1:][::-1]
        cathode = CONFIG_INFO['strips'][0][1:] + CONFIG_INFO['strips'][1][::-1]
        return mne.set_bipolar_reference(raw, anode, cathode)
    elif method == 'monopolar':
        return raw
    else:
        raise ValueError(f'Rereference method unacceptable, got {str(method)}, expect "monopolar" or "average" or "bipolar"')


def preprocessing(raw, reref_method='monopolar'):
    # cut by the first and last annotations
    annotation_onset, annotation_offset = raw.annotations.onset[0], raw.annotations.onset[-1]
    tmin, tmax = max(annotation_onset - 5., raw.times[0]), min(annotation_offset + 5, raw.times[-1])
    # rebuilt the raw
    # MNE的crop函数会导致annotation错乱，只能重建raw object
    new_annotations = mne.Annotations(onset=raw.annotations.onset - tmin,
                                  duration=raw.annotations.duration,
                                  description=raw.annotations.description)
    info = raw.info
    fs = info['sfreq']
    data = raw.get_data()
    # crop data
    data = data[..., int(tmin * fs):int(tmax * fs)]
    raw = mne.io.RawArray(data, info)
    raw.set_annotations(new_annotations)

    # do signal preprocessing
    raw.load_data()
    raw = reref(raw, reref_method)
    # high pass
    raw = raw.filter(1, None)
    # filter 50Hz
    raw = raw.notch_filter([50, 100, 150], trans_bandwidth=3, verbose=False)
    return raw


def reconstruct_events(events, fs, trial_duration=5, use_ori_events=False):
    """重构出事件序列中的单独运动事件
    Args:
        events (np.ndarray): 
        fs (float):
        trial_duration (float or None or dict): None means variable epoch length, dict means there are different trial durations for different trials 
        use_ori_events: skip deduplication
    """
    # Trial duration are fixed to be ? seconds.
    # extract trials
    if use_ori_events:
        events_new = events.copy()
    else:
        trials_ind_deduplicated = np.flatnonzero(np.diff(events[:, 2], prepend=0) != 0)
        events_new = events[trials_ind_deduplicated]
    if trial_duration is None:
        events_new[:-1, 1] = np.diff(events_new[:, 0])
        events_new[-1, 1] = events[-1, 0] - events_new[-1, 0]
    elif isinstance(trial_duration, dict):
        for e in trial_duration.keys():
            events_new[events_new[:, 2] == e, 1] = int(trial_duration[e] * fs)
    else:
        events_new[:, 1] = int(trial_duration * fs)
    return events_new


def load_sessions(data_root, session_names: dict, reref_method='monopolar'):
    # return raws for different finger models on an interleaved manner
    raw_cnt = sum(len(session_names[k]) for k in session_names)
    raws = []
    i = 0
    while i < raw_cnt:
        for finger_model in session_names.keys():
            try:
                s = session_names[finger_model].pop(0)
                i += 1
            except IndexError:
                continue
            # load raw
            raw = load_neuracle(os.path.join(data_root, s))
            # preprocess raw
            raw = preprocessing(raw, reref_method)
            # append list
            raws.append((finger_model, raw))
    return raws  


def load_neuracle(data_dir, data_type='ecog'):
    """
    neuracle file loader
    :param 
        data_dir: root data dir for the experiment
        sfreq: 
        data_type: 
    :return:
        raw: mne.io.RawArray
    """
    f = {
        'data': os.path.join(data_dir, 'data.bdf'),
        'evt': os.path.join(data_dir, 'evt.bdf'),
        'info': os.path.join(data_dir, 'recordInformation.json')
    }
    # read json
    with open(f['info'], 'r') as json_file:
        record_info = json.load(json_file)
    start_time_point = record_info['DataFileInformations'][0]['BeginTimeStamp']
    sfreq = record_info['SampleRate']

    # read data
    f_data = pyedflib.EdfReader(f['data'])
    ch_names = f_data.getSignalLabels()
    data = np.array([f_data.readSignal(i) for i in range(f_data.signals_in_file)]) * 1e-6  # to Volt

    info = mne.create_info(ch_names, sfreq, [data_type] * len(ch_names))
    raw = mne.io.RawArray(data, info)

    # read event
    try:
        f_evt = pyedflib.EdfReader(f['evt'])
        onset, duration, content = f_evt.readAnnotations()
        onset = np.array(onset) - start_time_point * 1e-3  # correct by start time point
        onset = (onset * sfreq).astype(np.int64)
        try:
            content = content.astype(np.int64)  # use original event code
        except ValueError:
            event_mapping = {c: i + 1 for i, c in enumerate(np.unique(content))}
            content = [event_mapping[i] for i in content]

        duration = (np.array(duration) * sfreq).astype(np.int64) 

        events = np.stack((onset, duration, content), axis=1)
        
        annotations = mne.annotations_from_events(events, sfreq)
        raw.set_annotations(annotations)
    except OSError:
        pass

    return raw