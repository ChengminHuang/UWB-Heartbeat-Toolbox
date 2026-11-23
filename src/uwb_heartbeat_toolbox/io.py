

import json
import numpy as np
from .models import RadarData

class RadarJsonReader:
    """负责读取雷达 JSON，并把 CIR 原始字节转成复数矩阵
    """

    def __init__(self, filename, empty_filename=None):
        self.filename = filename
        self.empty_filename = empty_filename

    def load(self):
        data, cir_complex = self._load_cir_complex_from_json(self.filename)
        taps = int(data["RX_RADAR_CIR_CONFIGURATION"]["CIR_TAPS"])

        rx_enbl = data["RX_RADAR_CONTROL_CONFIGURATION"]["RX_ENBL"]
        if rx_enbl == 2:
            n_0 = int(data["RADAR_NOISE_SUPPRESSION_CONFIGURATION"]["RNS_RX2"]["SELF_INTERFERENCE_TAP"])
        else:
            n_0 = int(data["RADAR_NOISE_SUPPRESSION_CONFIGURATION"]["RNS_RX1"]["SELF_INTERFERENCE_TAP"])

        step_time = data["CMD_RADAR_APPLICATION_CONFIG"]["FRAME_INTERVAL"] * 1e-9
        fs = 1.0 / step_time
        axis_time = np.arange(cir_complex.shape[0]) * step_time
        range_axis_m = (np.arange(taps) - n_0) * 0.15

        empty_cir_complex = None
        if self.empty_filename:
            empty_data, empty_cir_complex = self._load_cir_complex_from_json(self.empty_filename)
            empty_taps = int(empty_data["RX_RADAR_CIR_CONFIGURATION"]["CIR_TAPS"])
            if empty_taps != taps:
                raise ValueError(f"empty CIR_TAPS={empty_taps}, current CIR_TAPS={taps}")

        print("Data loaded")
        print("cir_complex shape =", cir_complex.shape)
        print("fs =", f"{fs:.3f} Hz")
        print("n_0 =", n_0)
        print("RX_ENBL =", rx_enbl)
        if empty_cir_complex is not None:
            print("empty_cir_complex shape =", empty_cir_complex.shape)

        return RadarData(
            raw_json=data,
            cir_complex=cir_complex,
            empty_cir_complex=empty_cir_complex,
            fs=fs,
            step_time=step_time,
            axis_time=axis_time,
            taps=taps,
            n_0=n_0,
            range_axis_m=range_axis_m,
        )

    def _load_cir_complex_from_json(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_frames = data["cir_data (decimal)"]
        taps = int(data["RX_RADAR_CIR_CONFIGURATION"]["CIR_TAPS"])
        cir_complex = np.zeros((len(raw_frames), taps), dtype=np.complex64)

        # 每个 tap 用 4 个字节表示：前 2 个字节是实部，后 2 个字节是虚部，都是小端有符号整数。
        for m, frame in enumerate(raw_frames):
            if len(frame) != taps * 4:
                raise ValueError(f"{json_path} frame {m}: length={len(frame)}, expected={taps * 4}")
            for k in range(taps):
                i0 = 4 * k
                real = int.from_bytes(bytes(frame[i0 : i0 + 2]), byteorder="little", signed=True)
                imag = int.from_bytes(bytes(frame[i0 + 2 : i0 + 4]), byteorder="little", signed=True)
                cir_complex[m, k] = real + 1j * imag

        return data, cir_complex
