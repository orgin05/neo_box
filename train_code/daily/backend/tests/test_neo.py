import unittest
import time
import numpy as np
from device.trigger_box import TriggerNeuracle
from device.data_client import NeuracleDataClient
from settings.config import settings


trigger_port = 'COM6'


class TestNeo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_info = settings.CONFIG_INFO
        cls.buffer_len = 1
        cls.receiver = NeuracleDataClient(n_channel=len(config_info['channel_labels']), 
                                          samplerate=config_info['sample_rate'],
                                          buffer_len=cls.buffer_len)
        cls.trigger = TriggerNeuracle(port=trigger_port)
    
    @classmethod
    def tearDownClass(cls) -> None:
        cls.receiver.close()
        return super().tearDownClass()
    
    def test_is_active(self):
        self.assertTrue(self.receiver.is_active())

    def test_get_data(self):
        time.sleep(1)
        fs, event, data = self.receiver.get_trial_data(clear=True)
        self.assertTrue(data.shape[1] == settings.CONFIG_INFO['sample_rate'] * self.buffer_len)
        self.assertTrue(data.shape[0] == len(settings.CONFIG_INFO['channel_labels']) - 1)
        self.assertTrue(event.size == 0)
    
    def test_highpass(self):
        time.sleep(5)
        fs, event, data = self.receiver.get_trial_data(clear=True)
        self.assertTrue(np.mean(data) < 5e-5)
    
    def test_send_trigger_and_receive(self):
        time.sleep(1)
        for i in range(5):
            self.trigger.send_trigger(i+1)
            time.sleep(0.1)
        fs, event, data = self.receiver.get_trial_data(clear=True)
        print(event.shape)
        self.assertEqual(event.shape[0], 5)
        self.assertTrue(np.allclose(event[:, 2], np.arange(1, 6)))
        self.assertTrue(data.shape[1] == settings.CONFIG_INFO['sample_rate'] * self.buffer_len)


    