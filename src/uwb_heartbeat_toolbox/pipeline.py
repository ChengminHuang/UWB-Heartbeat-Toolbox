

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .models import FrequencyBands, WindowResult
from .io import RadarJsonReader
from .signal_ops import SignalTools
from .gates import RangeGateSelector
from .rls import RespirationProcessor
from .heartbeat import HeartbeatProcessor
from .plotting import CumulativePlotter

class BreatheAndHeartSlidingWindowed:
    """滑动窗口总控类。
    1. 读雷达数据；
    2. 每个窗口选择距离门；
    3. 从主门取呼吸+心跳总相位；
    4. 提取呼吸；
    5. 扣除呼吸谐波后提取心跳；
    6. 对呼吸和心跳分别 OLA 累积；
    7. 动态图只显示当前窗口；结束后绘制 OLA 总结果和逐窗频率曲线。
    """

    def __init__(
        self,
        filename,
        preferred_range_m=(0.75, 7.65),
        empty_filename=None,
        window_sec=8.0,
        hop_sec=0.5,
        plot_every=1,
        detection_threshold=2000.0,
        vmd_k_options=(5,),
        vmd_alpha_options=(1000,),
        vmd_plot_dir=None,
        vmd_plot_format="tif",
        vmd_plot_show=False,
        vmd_plot_normalize=False,
        complex_var_threshold=None,
        resp_null_amplitude=0.01,
        heart_null_amplitude=0.001,
    ):
        self.filename = filename
        self.empty_filename = empty_filename
        self.preferred_range_m = preferred_range_m
        self.window_sec = float(window_sec)
        self.hop_sec = float(hop_sec)
        self.plot_every = int(plot_every)
        self.detection_threshold = float(detection_threshold)
        self.complex_var_threshold = (
            float(complex_var_threshold)
            if complex_var_threshold is not None
            else float(detection_threshold)
        )
        self.resp_null_amplitude = float(resp_null_amplitude)
        self.heart_null_amplitude = float(heart_null_amplitude)

        self.vmd_plot_dir = Path(vmd_plot_dir) if vmd_plot_dir else None
        self.vmd_plot_format = str(vmd_plot_format).lstrip(".").lower()
        self.vmd_plot_show = bool(vmd_plot_show)
        self.vmd_plot_normalize = bool(vmd_plot_normalize)
        self.vmd_plot_color = "#0072B2"

        self.bands = FrequencyBands()
        self.radar = RadarJsonReader(filename, empty_filename=empty_filename).load()
        self._check_sampling_rate()

        self.fs = self.radar.fs
        self.step_time = self.radar.step_time
        self.axis_time = self.radar.axis_time
        self.range_axis_m = self.radar.range_axis_m
        self.cir_complex = self.radar.cir_complex

        self.tools = SignalTools(self.radar.fs)
        self.range_selector = RangeGateSelector(
            self.radar,
            self.tools,
            self.bands,
            preferred_range_m=preferred_range_m,
            detection_threshold=detection_threshold,
        )
        self.respiration = RespirationProcessor(self.tools, self.bands, use_rls=False)
        self.heartbeat = HeartbeatProcessor(
            self.tools,
            self.bands,
            k=int(vmd_k_options[0]) if vmd_k_options else 5,
            alpha=float(vmd_alpha_options[0]) if vmd_alpha_options else 1000.0,
        )
        self.plotter = CumulativePlotter(self.radar.axis_time, self.tools, self.bands, self.heartbeat)

        self.total_sum = None
        self.resp_sum = None
        self.heart_sum = None
        self.ola_weight = None
        self.total_signal = None
        self.resp_signal = None
        self.heart_signal = None

        self.window_scores = []
        self.window_raw_bins = []
        self.window_selected_bins = []
        self.window_centers = []
        self.window_results = []

        self.last_vmd_modes = None
        self.last_imf_scores = None
        self.last_selected_imfs = []

    def run(self):
        num_frames = self.radar.cir_complex.shape[0]
        window_size = int(round(self.window_sec * self.radar.fs))
        hop_size = int(round(self.hop_sec * self.radar.fs))
        if window_size < 16:
            raise ValueError("window_sec is too short")
        if hop_size <= 0:
            raise ValueError("hop_sec must be positive")
        if window_size > num_frames:
            raise ValueError("window_sec is longer than the input signal")

        overlap_len = max(0, window_size - hop_size)
        ola_window = np.hanning(window_size)
        if np.max(ola_window) <= 0:
            ola_window = np.ones(window_size)

        self._reset_accumulators(num_frames)
        self.range_selector.reset_state()
        self.plotter.setup_dynamic()

        starts = list(range(0, num_frames - window_size + 1, hop_size))
        if not starts:
            raise ValueError("no valid sliding window starts were generated")

        prev_total_window = None
        prev_resp_window = None
        prev_heart_window = None

        for win_idx, start in enumerate(starts):
            end = start + window_size
            center_time = (start + end - 1) * 0.5 * self.radar.step_time
            cir_win = self.radar.cir_complex[start:end, :]

            # 1. 选择当前窗口的主距离门。
            gate = self.range_selector.select(cir_win)
            gate_var_score = float(np.nanmax(gate.score)) if len(gate.score) else 0.0
            gate_var_threshold = float(self.complex_var_threshold)
            gate_status = "ok" if gate_var_score >= gate_var_threshold else "low_complex_variance"

            resp_energy = 0.0
            heart_energy = 0.0
            resp_energy_threshold = self._reference_sine_energy(window_size, self.resp_null_amplitude)
            heart_energy_threshold = self._reference_sine_energy(window_size, self.heart_null_amplitude)
            resp_status = "ok"
            heart_status = "ok"

            if gate_status != "ok":
                # 当前窗口复数方差不足，直接判为无有效生命体征。
                total_window = np.zeros(window_size, dtype=np.float64)
                resp_window = np.zeros(window_size, dtype=np.float64)
                heart_window = np.zeros(window_size, dtype=np.float64)
                resp_freq = 0.0
                heart_freq = 0.0
                total_action = "zero_gate"
                resp_action = "zero_gate"
                heart_action = "zero_gate"
                resp_status = "zero_by_gate"
                heart_status = "zero_by_gate"
                heart_result = self._empty_heart_result(window_size)
            else:
                # 2. 只取主门 selected_bin 的相位，不再融合相邻门。
                total_window = self.range_selector.extract_main_gate_signal(cir_win, gate.selected_bin)
                total_window, total_corr, total_action = self.tools.match_sign(prev_total_window, total_window, overlap_len)
                total_window = total_window - np.mean(total_window)

                # 3. 从主门总相位里提取呼吸波形和呼吸频率。
                resp_window, resp_freq, _ = self.respiration.extract(total_window)
                resp_energy = self._signal_energy(resp_window)

                if resp_energy < resp_energy_threshold:
                    # 呼吸能量低于“无呼吸”标准正弦能量，当前窗口呼吸置零。
                    resp_window = np.zeros_like(resp_window)
                    resp_freq = 0.0
                    resp_action = "zero_resp_energy"
                    resp_status = "low_energy"
                else:
                    resp_window, resp_corr, resp_action = self.tools.match_sign(prev_resp_window, resp_window, overlap_len)
                    resp_status = "ok"

                # 4. 扣除呼吸谐波，得到更适合 VMD 的心跳输入。
                # 若呼吸被置零，则 resp_freq=0，谐波拟合不会引入呼吸基频项。
                resp_fit, residual, used_harmonics = self.respiration.fit_and_remove_harmonics(total_window, resp_freq)
                heart_input, heart_result = self.heartbeat.extract(residual, resp_freq)
                self._maybe_plot_vmd_components(
                    win_idx=win_idx,
                    start=start,
                    end=end,
                    window_size=window_size,
                    heart_result=heart_result,
                )
                heart_window = heart_result["reconstructed"] - np.mean(heart_result["reconstructed"])
                heart_freq = float(heart_result["heart_freq"])
                heart_energy = self._signal_energy(heart_window)

                if heart_energy < heart_energy_threshold:
                    # 心跳能量低于“无心跳”标准正弦能量，当前窗口心跳置零。
                    heart_window = np.zeros_like(heart_window)
                    heart_freq = 0.0
                    heart_action = "zero_heart_energy"
                    heart_status = "low_energy"
                else:
                    heart_window, heart_corr, heart_action = self.tools.match_sign(prev_heart_window, heart_window, overlap_len)
                    heart_status = "ok"

            # 5. OLA 累积。这里的 OLA 是跨窗口拼接，不是距离门融合。
            self.total_sum[start:end] += total_window * ola_window
            self.resp_sum[start:end] += resp_window * ola_window
            self.heart_sum[start:end] += heart_window * ola_window
            self.ola_weight[start:end] += ola_window

            # 6. 当前窗口输出只使用当前窗口内估计出的呼吸率和心率；
            #    低于方差/能量阈值的窗口直接输出 0。
            result = self._record_window_result(
                win_idx,
                start,
                end,
                center_time,
                gate,
                resp_freq,
                heart_freq,
                np.nan,
                np.nan,
                heart_result,
                total_action,
                resp_action,
                heart_action,
                gate_var_score,
                gate_var_threshold,
                resp_energy,
                resp_energy_threshold,
                heart_energy,
                heart_energy_threshold,
                gate_status,
                resp_status,
                heart_status,
            )
            self._print_window_result(result)

            if win_idx % self.plot_every == 0:
                self.plotter.update_dynamic(
                    win_idx=win_idx,
                    window_start_sec=float(start / self.radar.fs),
                    resp_window=resp_window,
                    heart_window=heart_window,
                    resp_freq=resp_freq,
                    heart_freq=heart_freq,
                )

            prev_total_window = total_window.copy()
            prev_resp_window = resp_window.copy()
            prev_heart_window = heart_window.copy()
            self.last_vmd_modes = heart_result["modes"]
            self.last_imf_scores = heart_result["scores"]
            self.last_selected_imfs = heart_result["selected_indices"]

        self.total_signal = self.tools.normalize_ola(self.total_sum, self.ola_weight)
        self.resp_signal = self.tools.normalize_ola(self.resp_sum, self.ola_weight)
        self.heart_signal = self.tools.normalize_ola(self.heart_sum, self.ola_weight)

        plt.ioff()
        self.plot_range_gate_curve()
        self.plot_final_ola_results()
        self.plot_window_rate_curve()
        return self.resp_signal, self.heart_signal

    def _maybe_plot_vmd_components(self, win_idx, start, end, window_size, heart_result):
        """
        Save one VMD IMF figure only for non-overlapping window positions.

        Example:
            window_sec = 10 s, hop_sec = 1 s
            plotted windows: 0-10 s, 10-20 s, 20-30 s, ...
            skipped windows: 1-11 s, 2-12 s, ...
        """

        if self.vmd_plot_dir is None:
            return

        # Only plot windows whose start index is an integer multiple of window_size.
        # This implements 0-window_sec, window_sec-2*window_sec, ...
        if int(start) % int(window_size) != 0:
            return

        self.vmd_plot_dir.mkdir(parents=True, exist_ok=True)

        start_sec = float(start / self.radar.fs)
        end_sec = float(end / self.radar.fs)

        start_tag = f"{start_sec:07.2f}".replace(".", "p")
        end_tag = f"{end_sec:07.2f}".replace(".", "p")

        if self.vmd_plot_format:
            save_path = self.vmd_plot_dir / f"vmd_imfs_win{win_idx:03d}_{start_tag}-{end_tag}s.{self.vmd_plot_format}"
        else:
            save_path = self.vmd_plot_dir / f"vmd_imfs_win{win_idx:03d}_{start_tag}-{end_tag}s"

        self.heartbeat.plot_vmd_imfs_for_window(
            vmd_result=heart_result,
            window_start_sec=start_sec,
            window_end_sec=end_sec,
            save_path=save_path,
            show=self.vmd_plot_show,
            normalize=self.vmd_plot_normalize,
            color=self.vmd_plot_color,
        )

    def plot_range_gate_curve(self):
        if not self.window_results:
            raise ValueError("run() must be called first")
        self.plotter.plot_range_gate_curve(
            centers=np.asarray(self.window_centers),
            selected_bins=np.asarray(self.window_selected_bins),
            raw_bins=np.asarray(self.window_raw_bins),
            range_axis_m=self.radar.range_axis_m,
            n_0=self.radar.n_0,
        )

    def plot_final_ola_results(self):
        if self.resp_signal is None or self.heart_signal is None:
            raise ValueError("run() must be called first")
        self.plotter.plot_final_ola_results(
            self.resp_signal,
            self.heart_signal,
            save_path=None,
            show=True,
        )

    def plot_window_rate_curve(self):
        if not self.window_results:
            raise ValueError("run() must be called first")
        self.plotter.plot_window_rate_curve(
            centers=np.asarray(self.window_centers),
            resp_freqs=np.asarray([r.resp_freq for r in self.window_results], dtype=np.float64),
            heart_freqs=np.asarray([r.heart_freq for r in self.window_results], dtype=np.float64),
            save_path=None,
            show=True,
        )

    def _current_cumulative_state(self, current_end):
        resp_current = self.tools.normalize_ola(self.resp_sum, self.ola_weight)
        heart_current = self.tools.normalize_ola(self.heart_sum, self.ola_weight)
        current_end = min(int(current_end), len(resp_current), len(heart_current))

        resp_seg = resp_current[:current_end]
        heart_seg = heart_current[:current_end]
        resp_freq = self._freq_if_long_enough(resp_seg, self.bands.resp)
        heart_freq = self._heart_freq_if_long_enough(heart_seg, resp_freq)
        return resp_current, heart_current, current_end, resp_freq, heart_freq

    def _freq_if_long_enough(self, x, band):
        # 累计信号太短时频谱不稳定，这里等至少 4 秒后再报频率。
        if len(x) < max(4, int(round(4.0 * self.radar.fs))) or np.std(x) < 1e-12:
            return np.nan
        freq, _, _, _ = self.tools.dominant_freq(x, band)
        return freq


    def _heart_freq_if_long_enough(self, x, resp_freq):
        if len(x) < max(4, int(round(4.0 * self.radar.fs))) or np.std(x) < 1e-12:
            return np.nan
        freq, _, _, _ = self.tools.dominant_freq(x, self.bands.heart)
        return freq

    def _record_window_result(
        self,
        win_idx,
        start,
        end,
        center_time,
        gate,
        resp_freq,
        heart_freq,
        cum_resp_freq,
        cum_heart_freq,
        heart_result,
        total_action,
        resp_action,
        heart_action,
        gate_var_score,
        gate_var_threshold,
        resp_energy,
        resp_energy_threshold,
        heart_energy,
        heart_energy_threshold,
        gate_status,
        resp_status,
        heart_status,
    ):
        selected_range_m = float(self.radar.range_axis_m[gate.selected_bin])
        result = WindowResult(
            window_index=int(win_idx),
            start_sec=float(start / self.radar.fs),
            end_sec=float(end / self.radar.fs),
            center_sec=float(center_time),
            raw_bin=int(gate.raw_bin),
            selected_bin=int(gate.selected_bin),
            selected_range_m=selected_range_m,
            resp_freq=float(resp_freq),
            heart_freq=float(heart_freq),
            cum_resp_freq=float(cum_resp_freq),
            cum_heart_freq=float(cum_heart_freq),
            selected_imfs=[int(v) for v in heart_result["selected_indices"]],
            total_sign_action=total_action,
            resp_sign_action=resp_action,
            heart_sign_action=heart_action,
            gate_var_score=float(gate_var_score),
            gate_var_threshold=float(gate_var_threshold),
            resp_energy=float(resp_energy),
            resp_energy_threshold=float(resp_energy_threshold),
            heart_energy=float(heart_energy),
            heart_energy_threshold=float(heart_energy_threshold),
            gate_status=str(gate_status),
            resp_status=str(resp_status),
            heart_status=str(heart_status),
        )

        self.window_results.append(result)
        self.window_scores.append(gate.score)
        self.window_raw_bins.append(gate.raw_bin)
        self.window_selected_bins.append(gate.selected_bin)
        self.window_centers.append(center_time)
        return result

    def _print_window_result(self, result):
        print(
            f"window {result.window_index:03d}: "
            f"{result.start_sec:6.2f}-{result.end_sec:6.2f}s, "
            f"bin={result.selected_bin:3d} ({result.selected_range_m:.2f} m), "
            f"resp={result.resp_freq:.4f} Hz/{result.resp_freq * 60.0:.2f} BPM, "
            f"heart={result.heart_freq:.4f} Hz/{result.heart_freq * 60.0:.2f} BPM, "
            f"gate_score={result.gate_var_score:.2f}/{result.gate_var_threshold:.2f}, "
            f"E_resp={result.resp_energy:.4g}/{result.resp_energy_threshold:.4g}, "
            f"E_heart={result.heart_energy:.4g}/{result.heart_energy_threshold:.4g}, "
            f"status(gate/resp/heart)="
            f"{result.gate_status}/{result.resp_status}/{result.heart_status}, "
            f"imfs={result.selected_imfs}, "
            f"sign(total/resp/heart)="
            f"{result.total_sign_action}/{result.resp_sign_action}/{result.heart_sign_action}"
        )

    @staticmethod
    def _signal_energy(x):
        """窗口能量定义为离散信号平方和。"""
        x = np.asarray(x, dtype=np.float64)
        if len(x) == 0:
            return 0.0
        return float(np.sum(x ** 2))

    @staticmethod
    def _reference_sine_energy(n, amplitude):
        """幅值为 amplitude 的标准正弦在当前窗口长度下的参考能量。

        这里使用一个完整周期的离散正弦：
            x[k] = A sin(2 pi k / n), k = 0, ..., n-1

        因此能量随窗口采样点数 n 变化，近似为 n * A^2 / 2。
        """
        n = int(n)
        amplitude = float(amplitude)
        if n <= 0:
            return 0.0
        k = np.arange(n, dtype=np.float64)
        ref = amplitude * np.sin(2.0 * np.pi * k / n)
        return float(np.sum(ref ** 2))

    def _empty_heart_result(self, n):
        """方差门控失败时构造空 VMD 结果，保证后续记录和绘图逻辑一致。"""
        n = int(n)
        k = int(getattr(self.heartbeat, "k", 1))
        return {
            "K": k,
            "alpha": float(getattr(self.heartbeat, "alpha", 0.0)),
            "modes": np.zeros((k, n), dtype=np.float64),
            "omega": np.zeros((1, k), dtype=np.float64),
            "selected_indices": [],
            "reconstructed": np.zeros(n, dtype=np.float64),
            "scores": [],
            "heart_freq": 0.0,
            "quality": 0.0,
        }

    def _reset_accumulators(self, num_frames):
        self.total_sum = np.zeros(num_frames, dtype=np.float64)
        self.resp_sum = np.zeros(num_frames, dtype=np.float64)
        self.heart_sum = np.zeros(num_frames, dtype=np.float64)
        self.ola_weight = np.zeros(num_frames, dtype=np.float64)

        self.window_scores = []
        self.window_raw_bins = []
        self.window_selected_bins = []
        self.window_centers = []
        self.window_results = []

    def _check_sampling_rate(self):
        nyquist = 0.5 * self.radar.fs
        if self.bands.total[1] >= nyquist:
            raise ValueError(f"fs={self.radar.fs:.3f} Hz is too low for high cutoff {self.bands.total[1]} Hz")
