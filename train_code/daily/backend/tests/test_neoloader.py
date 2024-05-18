import unittest
from dataloaders import neo
import mne
import numpy as np


class TestDataloader(unittest.TestCase):
    def test_load_sample_data(self):
        root_path = './tests/data'
        sessions = {'flex': ['1']}

        raw, event_id = neo.raw_loader(root_path, sessions, upsampled_epoch_length=1.)
        events, event_id = mne.events_from_annotations(raw, event_id=event_id)
        events, events_cnt = np.unique(events[:, -1], return_counts=True)
        self.assertTrue(np.allclose(events_cnt, (75, 75)))

    def test_load_session(self):
        root_path = './tests/data'
        sessions = {'flex': ['1', '3'], 'ball': ['2']}
        raws = neo.load_sessions(root_path, sessions)
        # test if interleaved
        sess_f = tuple(f for f, r in raws)
        self.assertEqual(len(raws), 3)
        self.assertTupleEqual(sess_f, ('flex', 'ball', 'flex'))


    def test_event_parser(self):
        # fixed length
        fs = 100
        test_event = np.array([[0, 0, 4], [100, 0, 4], [600, 0, 3], [700, 0, 3], [1000, 0, 4], [1100, 0, 4]])
        gt = np.array([[0, 400, 4], [600, 400, 3], [1000, 400, 4]])
        ret = neo.reconstruct_events(test_event, fs, trial_duration=4)
        self.assertTrue(np.allclose(ret, gt))
        # duration as dict
        gt = np.array([[0, 400, 4], [600, 200, 3], [1000, 400, 4]])
        trial_duration = {4: 4., 3: 2.}
        ret = neo.reconstruct_events(test_event, fs, trial_duration=trial_duration)
        self.assertTrue(np.allclose(ret, gt))
        # varing length
        gt = np.array([[0, 600, 4], [600, 400, 3], [1000, 100, 4]])
        ret = neo.reconstruct_events(test_event, fs, trial_duration=None)
        self.assertTrue(np.allclose(ret, gt))

        # use ori
        test_event = np.array([[0, 0, 4], [100, 0, 4], [600, 0, 3], [700, 0, 3], [1000, 0, 4], [1100, 0, 4]])
        gt = np.array([[0, 400, 4], [100, 400, 4], [600, 400, 3], [700, 400, 3], [1000, 400, 4], [1100, 400, 4]])
        ret = neo.reconstruct_events(test_event, fs, trial_duration=4, use_ori_events=True)
        self.assertTrue(np.allclose(ret, gt))

