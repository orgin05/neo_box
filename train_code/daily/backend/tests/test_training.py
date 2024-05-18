import os
import training
import unittest
import joblib
from glob import glob
from sklearn.linear_model import LogisticRegression
from dataloaders import neo
import bci_core.utils as bci_utils
from bci_core.feature_extractors import FeatExtractor
from bci_core.pipeline import riemann_model_builder, baseline_model_builder
import shutil
from sklearn.utils.validation import check_is_fitted
from sklearn.pipeline import Pipeline


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root_path = './tests/data'
        sessions = {'flex': ['1', '2']}

        raw, cls.event_id = neo.raw_loader(root_path, sessions)
        cls.raw = raw
    
    def test_training_baseline(self):
        model = training.train_model(self.raw, self.event_id, model_type='baseline')
        check_is_fitted(model[-1])
    
    def test_training_riemann(self):
        model = training.train_model(self.raw, self.event_id, model_type='riemann')
        check_is_fitted(model[-1])

    def test_saver(self):
        feat_ext, embedder = riemann_model_builder(1000, 9, lf_bands=[(15, 30), [30, 45]], hg_bands=[(55, 95), (105, 145)])
        clf = LogisticRegression()

        event_id = {'1': 5, '0': 3}
        bci_utils.model_saver([feat_ext, embedder, clf], './tests/data', 'baseline', 'f77cbe10a8de473992542e9f4e913a66', event_id)
        self.assertTrue(os.path.isdir(os.path.join('./tests/data', 'f77cbe10a8de473992542e9f4e913a66')))

        model_file = glob(os.path.join('./tests/data', 
                                       'f77cbe10a8de473992542e9f4e913a66', 
                                       '*.pkl'))
        
        self.assertEqual(len(model_file), 1)

        name = os.path.normpath(model_file[0]).split(os.sep)
        class_name, events, date = name[-1].split('_')
        print(class_name, events, date)
        self.assertTrue(class_name == 'baseline')
        self.assertTrue(events == '0+1')
        # load model
        feat, embedder, model_base = joblib.load(model_file[0])
        self.assertTrue(isinstance(feat, FeatExtractor))
        self.assertTrue(isinstance(embedder, Pipeline))
        self.assertTrue(isinstance(model_base, LogisticRegression))

        shutil.rmtree(os.path.join('./tests/data', 'f77cbe10a8de473992542e9f4e913a66'))
