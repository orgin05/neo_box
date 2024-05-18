import unittest
import time
from device.fubo_pneumatic_finger import FuboPneumaticFingerClient, get_serial_ports

PORT = "COM3"
init_params = {"port": PORT}


class TestPeripheralHand(unittest.TestCase):
    def test_get_ports_from_computer_success(self):
        ports = get_serial_ports()
        self.assertTrue(len(ports) > 0)

    def test_client_init_success(self):
        client = FuboPneumaticFingerClient(init_params)
        self.assertTrue(client.is_connected)
        client.close()

    def test_client_close_success(self):
        client = FuboPneumaticFingerClient(init_params)
        client.close()
        self.assertFalse(client.is_connected)

    def test_start_extend_success(self):
        client = FuboPneumaticFingerClient(init_params)
        client.start('extend')
        time.sleep(3)
        client.close()

    def test_start_operate_success(self):
        client = FuboPneumaticFingerClient(init_params)
        client.start('flex')
        time.sleep(3)
        client.close()