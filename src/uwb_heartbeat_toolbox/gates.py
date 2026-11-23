

import numpy as np
from .models import RangeGateResult

class RangeGateSelector:
    """滑窗内距离门选择和主门相位提取类。
    - 当前窗口每个距离门求方差；
    - 与空场 reference 方差做差；
    - 在 ROI 内取最大分数作为候选目标；
    - 用 decision_counter 做简单稳定确认。

    窗口内融合相邻门，只提取 selected_bin 主门信号。
    """

    def __init__(
        self,
        radar_data,
        tools,
        bands,
        preferred_range_m=(0.75, 7.65),
        detection_threshold=2000.0,
        use_savgol=False,
    ):
        self.radar = radar_data
        self.tools = tools
        self.bands = bands
        self.preferred_range_m = preferred_range_m
        self.detection_threshold = float(detection_threshold)
        self.use_savgol = use_savgol

        self.decision_counter_min = 0
        self.decision_counter_max = 6
        self.reset_state()
        self.ref_var = self._prepare_reference()

    def reset_state(self):
        self.decision_counter = 3
        self.last_target_tap = 0
        self.target_detected = False
        self.confirmed_target_tap = self.radar.n_0

    def select(self, cir_win):
        cur_var = np.var(cir_win, axis=0)
        score = np.abs(cur_var - self.ref_var)

        # 自干扰 tap 和它之前的区域不参与目标选择。
        score[: self.radar.n_0 + 1] = 0.0

        if self.preferred_range_m is not None:
            rmin, rmax = self.preferred_range_m
            roi_mask = (self.radar.range_axis_m >= rmin) & (self.radar.range_axis_m <= rmax)
            score = np.where(roi_mask, score, 0.0)

        raw_bin = int(np.argmax(score)) if score.max() > self.detection_threshold else 0
        valid_target = raw_bin > self.radar.n_0
        same_target = valid_target and (raw_bin == self.last_target_tap)

        if same_target:
            self.decision_counter = min(self.decision_counter + 1, self.decision_counter_max)
        else:
            self.decision_counter = max(self.decision_counter - 1, self.decision_counter_min)
            self.last_target_tap = raw_bin if valid_target else 0

        if self.decision_counter > 3 and same_target:
            self.target_detected = True
            self.confirmed_target_tap = self.last_target_tap
        elif self.decision_counter < 2:
            self.target_detected = False
            self.confirmed_target_tap = self.radar.n_0

        selected_bin = self.confirmed_target_tap if self.target_detected else max(self.radar.n_0, raw_bin)
        cands = np.arange(selected_bin - 1, selected_bin + 2)
        cands = cands[(cands >= self.radar.n_0) & (cands < self.radar.taps)]
        if len(cands) == 0:
            cands = np.array([selected_bin], dtype=int)

        return RangeGateResult(
            score=score,
            raw_bin=int(raw_bin),
            selected_bin=int(selected_bin),
            candidate_bins=cands.astype(int),
            target_detected=bool(self.target_detected),
            decision_counter=int(self.decision_counter),
        )

    def extract_main_gate_signal(self, cir_win, selected_bin):
        """只提取主门相位，并滤成呼吸+心跳总带宽信号。"""

        raw_phase = np.angle(cir_win[:, selected_bin])
        phase_unwrap = np.unwrap(raw_phase)
        phase_detrended = self.tools.detrend_phase(phase_unwrap, use_savgol=self.use_savgol)
        return self.tools.bandpass(phase_detrended, self.bands.total[0], self.bands.total[1], order=3)

    def _prepare_reference(self):
        if self.radar.empty_cir_complex is not None:
            print("Reference variance uses empty data")
            return np.var(self.radar.empty_cir_complex, axis=0)

        ref_len = min(max(int(5 * self.radar.fs), 200), max(1, self.radar.cir_complex.shape[0] // 3))
        print(f"Reference variance uses the first {ref_len} frames")
        return np.var(self.radar.cir_complex[:ref_len, :], axis=0)
