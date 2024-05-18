import unittest
import os
import numpy as np
from glob import glob
import shutil

import mne

from bci_core import utils as ana_utils
from bci_core.online import model_loader
from training import train_model
from dataloaders import neo
from online_sim import simulation, _construct_model_event
from validation import val_by_epochs


class TestOnlineSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root_path = './tests/data'

        raw_train, cls.event_id = neo.raw_loader(root_path, {'flex': ['1']}, reref_method='bipolar')
        cls.raw_val, _ = neo.raw_loader(root_path, {'flex': ['2']}, 
                                        upsampled_epoch_length=None,
                                        reref_method='bipolar')
        
        # train with the first half
        model = train_model(raw_train, event_id=cls.event_id, model_type='baseline')
        ana_utils.model_saver(model, './tests/data/', 'baseline', 'test', cls.event_id)
        cls.model_path = glob(os.path.join('./tests/data/', 'test', '*.pkl'))[0]
    
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(os.path.join('./tests/data/', 'test'))
        return super().tearDownClass()
    
    def test_event_metric(self):
        event_gt = np.array([[0, 0, 0], [5, 0, 1], [7, 0, 0], [9, 0, 2]])
        event_pred = np.array([[1, 0, 0], [4, 0, 1], [6, 0, 1], [7, 0, 0], [10, 0, 1], [11, 0, 2]])
        fs = 1
        precision, recall, f1_score = ana_utils.event_metric(event_gt, event_pred, fs, ignore_event=(0,))
        self.assertEqual(f1_score, 2 / 3)
        self.assertEqual(precision, 1 / 2)
        self.assertEqual(recall, 1)

    def test_construct_event(self):
        seq_1 = [(1, -1), (2, -1), (3, -1), (4, 1)]
        seq_2 = [(1, 0), (2, 0), (4, 1)]
        gt = [[1, 0, 0], [4, 0, 1]]
        ret_ = _construct_model_event(seq_1, 1, start_cond=0)
        self.assertTrue(np.allclose(gt, ret_))
        ret_ = _construct_model_event(seq_2, 1, start_cond=0)
        self.assertTrue(np.allclose(gt, ret_))

    def test_sim(self):
        model = model_loader(self.model_path, 
                             state_change_threshold=0.7,
                             state_trans_prob=0.7)
        metric_hmm, metric_nohmm, figs = simulation(self.raw_val, self.event_id, model=model, epoch_length=1., step_length=0.1)
        figs[0].savefig('./tests/data/pred_hmm.pdf')
        figs[1].savefig('./tests/data/pred_naive.pdf')
        self.assertTrue(metric_hmm[-2] > 0.7)  # f1-score (with hmm)
        self.assertTrue(metric_nohmm[-2] < 0.4)  # f1-score (without hmm)
    
    def test_val_model(self):
        metrices, fig_conf = val_by_epochs(self.raw_val, self.model_path, self.event_id, 1.)
        fig_conf.savefig('./tests/data/conf.pdf')
        self.assertGreater(metrices[0], 0.85)
        self.assertGreater(metrices[1], 0.7)
        self.assertGreater(metrices[2], 0.7)


if __name__ == '__main__':
    unittest.main()