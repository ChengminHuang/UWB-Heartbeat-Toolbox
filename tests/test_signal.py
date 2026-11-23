import unittest
import numpy as np
from uwb_heartbeat_toolbox.signal_ops import SignalTools


class SignalToolsTest(unittest.TestCase):
    def test_fft_known_frequency(self):
        fs = 10.0
        t = np.arange(200) / fs
        x = np.sin(2.0 * np.pi * 1.0 * t)
        freq, amplitude = SignalTools(fs).fft_amp(x)
        self.assertAlmostEqual(float(freq[np.argmax(amplitude)]), 1.0)


if __name__ == "__main__":
    unittest.main()
