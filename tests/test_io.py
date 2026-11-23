import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from uwb_heartbeat_toolbox.io import RadarJsonReader


def four_bytes(real, imag):
    return list(int(real).to_bytes(2, "little", signed=True) + int(imag).to_bytes(2, "little", signed=True))


class RadarJsonReaderTest(unittest.TestCase):
    def test_decode_two_complex_taps(self):
        content = {
            "RX_RADAR_CIR_CONFIGURATION": {"CIR_TAPS": 2},
            "RX_RADAR_CONTROL_CONFIGURATION": {"RX_ENBL": 2},
            "RADAR_NOISE_SUPPRESSION_CONFIGURATION": {"RNS_RX2": {"SELF_INTERFERENCE_TAP": 0}},
            "CMD_RADAR_APPLICATION_CONFIG": {"FRAME_INTERVAL": 100000000},
            "cir_data (decimal)": [four_bytes(3, -4) + four_bytes(-7, 2),
                                   four_bytes(1, 0) + four_bytes(0, -1)],
        }
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.json"
            source.write_text(json.dumps(content), encoding="utf-8")
            record = RadarJsonReader(source).load()
        self.assertAlmostEqual(record.fs, 10)
        np.testing.assert_array_equal(record.cir_complex,
                                      np.array([[3-4j, -7+2j], [1+0j, 0-1j]]))


if __name__ == "__main__":
    unittest.main()
