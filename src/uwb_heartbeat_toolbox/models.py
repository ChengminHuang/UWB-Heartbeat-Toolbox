

from dataclasses import dataclass
import numpy as np

@dataclass
class FrequencyBands:
    """集中管理频带。"""

    resp: tuple = (0.134, 0.5)
    heart: tuple = (0.834, 2.2)

    @property
    def total(self):
        # 主门相位先保留呼吸加心跳的总带宽，后面再分离呼吸和心跳。
        return (self.resp[0], self.heart[1])


@dataclass
class RadarData:
    """雷达 JSON 解码后的核心数据和坐标。"""

    raw_json: dict
    cir_complex: np.ndarray
    empty_cir_complex: np.ndarray
    fs: float
    step_time: float
    axis_time: np.ndarray
    taps: int
    n_0: int
    range_axis_m: np.ndarray


@dataclass
class RangeGateResult:
    """一个滑窗内的距离门选择结果。"""

    score: np.ndarray
    raw_bin: int
    selected_bin: int
    candidate_bins: np.ndarray
    target_detected: bool
    decision_counter: int


@dataclass
class WindowResult:
    """每个滑窗最终输出的摘要，方便打印或后续分析。"""

    window_index: int
    start_sec: float
    end_sec: float
    center_sec: float
    raw_bin: int
    selected_bin: int
    selected_range_m: float
    resp_freq: float
    heart_freq: float
    cum_resp_freq: float
    cum_heart_freq: float
    selected_imfs: list
    total_sign_action: str
    resp_sign_action: str
    heart_sign_action: str
    gate_var_score: float
    gate_var_threshold: float
    resp_energy: float
    resp_energy_threshold: float
    heart_energy: float
    heart_energy_threshold: float
    gate_status: str
    resp_status: str
    heart_status: str
