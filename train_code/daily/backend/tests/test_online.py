import os
import shutil
import random
import bci_core.online as online
from bci_core.utils import model_saver
import training
from dataloaders import neo
from online_sim import DataGenerator
import unittest
import numpy as np
from glob import glob


class TestOnline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root_path = './tests/data'

        raw, event_id = neo.raw_loader(root_path, {'flex': ['1']}, reref_method='bipolar')
        
        model = training.train_model(raw, event_id, model_type='baseline')
        
        model_saver(model, root_path, 'baseline', 'f77cbe10a8de473992542e9f4e913a66', event_id)
        cls.model_root = os.path.join(root_path, 'f77cbe10a8de473992542e9f4e913a66')
        cls.model_path = glob(os.path.join(root_path, 'f77cbe10a8de473992542e9f4e913a66', '*.pkl'))[0]

        raw, event_id = neo.raw_loader(root_path, {'flex': ['1']}, reref_method='monopolar')
        cls.data_gen = DataGenerator(raw.info['sfreq'], raw.get_data())
    
    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.model_root)
        return super().tearDownClass()
    
    def test_step_feedback(self):
        model_hmm = online.model_loader(self.model_path)
        controller = online.Controller(0, model_hmm, reref_method='bipolar')
        rets = []
        for time, data in self.data_gen.loop():
            cls = controller.step_decision(data)
            rets.append(cls)
        self.assertTrue(np.allclose(np.unique(rets), [0, 3]))
    
    def test_virtual_feedback(self):
        controller = online.Controller(1, None)
        
        n_trial = 1000
        correct = 0
        for _ in range(n_trial):
            label = random.randint(0, 1)
            ret = controller.decision(None, label)
            if ret == label:
                correct += 1
        self.assertTrue(abs(correct / n_trial - 0.8) < 0.1)

        correct = 0
        for _ in range(n_trial):
            label = random.randint(0, 1)
            ret = controller.step_decision(None, label)
            if ret == label:
                correct += 1
        self.assertTrue(abs(correct / n_trial - 0.8) < 0.1)

    def test_real_feedback(self):
        model_hmm = online.model_loader(self.model_path)
        controller = online.Controller(0, model_hmm, reref_method='bipolar')
        rets = []
        for i, (time, data) in zip(range(300), self.data_gen.loop()):
            cls = controller.decision(data)
            rets.append(cls)
        self.assertTrue(np.allclose(np.unique(rets), [-1, 0, 3]))


class TestHMM(unittest.TestCase):
    def test_state_transfer(self):
        # binary
        probs = [[0.9, 0.1], [0.5, 0.5], [0.09, 0.91], [0.5, 0.5], [0.3, 0.7], [0.7, 0.3], [0.92,0.08]]
        true_state = [-1, -1, 1, -1, -1, -1, 0]
        model = online.HMMModel(transmat=None, n_classes=2, state_trans_prob=0.9, state_change_threshold=0.7, momentum=0.)
        states = []
        for p in probs:
            cur_state = model.update_state(p)
            states.append(cur_state)
        self.assertTrue(np.allclose(states, true_state))

        # triple
        probs = [[0.8, 0.1, 0.1], [0.01, 0.91, 0.09], [0.01, 0.08, 0.91], [0.5, 0.2, 0.3], [0.9, 0.05, 0.02], [0.01, 0.01, 0.98]]
        true_state = [-1, 1, -1, -1, 0, 2]
        model = online.HMMModel(transmat=None, n_classes=3, state_trans_prob=0.9, momentum=0.)
        states = []
        for p in probs:
            cur_state = model.update_state(p)
            states.append(cur_state)
        self.assertTrue(np.allclose(states, true_state))
    
    def test_momentum(self):
        # binary
        probs = [[0.9, 0.1], [0.5, 0.5], [0.09, 0.91], [0.01, 0.99], [0.3, 0.7], [0.7, 0.3], [0.92,0.08]]
        true_state = [-1, -1, -1, -1, 1, -1, -1]
        model = online.HMMModel(transmat=None, n_classes=2, state_trans_prob=0.9, state_change_threshold=0.7, momentum=0.5)
        states = []
        for p in probs:
            cur_state = model.update_state(p)
            states.append(cur_state)
        print(states)
        self.assertTrue(np.allclose(states, true_state))