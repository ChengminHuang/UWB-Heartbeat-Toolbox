

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal
from .vmdpy import VMD

class HeartbeatProcessor:
    """心跳处理类。
    1. 将扣除呼吸谐波后的残差滤到心跳频带；
    2. 单次 VMD 分解；
    3. 根据心跳成分自身特征建立模糊评分；
    4. 选择模糊综合评分最高的 IMF。
    """

    def __init__(
        self,
        tools,
        bands,
        k=5,
        alpha=1000.0,
        heart_peak_prefer_min_hz=1.0,
        heart_peak_prefer_max_hz=1.8,
        peak_search_min_relative_amp=0.15,
    ):
        self.tools = tools
        self.bands = bands
        self.k = int(k)
        self.alpha = float(alpha)
        self.heart_peak_prefer_min_hz = float(heart_peak_prefer_min_hz)
        self.heart_peak_prefer_max_hz = float(heart_peak_prefer_max_hz)
        self.peak_search_min_relative_amp = float(peak_search_min_relative_amp)

    def extract(self, residual, resp_freq=None):
        heart_input = self.tools.bandpass(residual, self.bands.heart[0], self.bands.heart[1], order=3)
        vmd_result = self._run_vmd_once(heart_input)
        return heart_input, vmd_result

    def dominant_heart_freq(self, x, resp_freq=None):
        freq, mag = self.tools.fft_amp(x)
        mask = (freq >= self.bands.heart[0]) & (freq <= self.bands.heart[1])
        if not np.any(mask):
            return np.nan, 0.0, freq, mag

        local_freq = freq[mask]
        local_mag = mag[mask]
        peak_indices, _ = signal.find_peaks(local_mag)

        if len(peak_indices) == 0:
            peak_indices = np.array([int(np.argmax(local_mag))])

        max_amp = float(np.max(local_mag)) + 1e-12
        valid_peaks = [
            idx for idx in peak_indices
            if local_mag[idx] >= self.peak_search_min_relative_amp * max_amp
        ]
        if not valid_peaks:
            valid_peaks = [int(np.argmax(local_mag))]

        best = None
        for idx in valid_peaks:
            peak_freq = float(local_freq[idx])
            peak_amp = float(local_mag[idx])

            freq_score = self._freq_prior_membership(peak_freq)
            amp_score = peak_amp / max_amp
            score = 0.65 * amp_score + 0.35 * freq_score

            item = (score, peak_freq, peak_amp)
            if best is None or item[0] > best[0]:
                best = item

        return float(best[1]), float(best[2]), freq, mag

    def _run_vmd_once(self, heart_input, tau=0, dc=0, init=0, tol=1e-5):
        x = np.asarray(heart_input, dtype=np.float64)
        x = x - np.mean(x)
        modes, _, omega = VMD(x, self.alpha, tau, self.k, dc, init, tol)

        scores = []
        for i in range(modes.shape[0]):
            item = self._score_imf_fuzzy(modes[i])
            if item is None:
                continue
            item["imf_index"] = int(i)
            scores.append(item)

        if not scores:
            heart_freq, _, _, _ = self.dominant_heart_freq(x)
            return {
                "K": self.k,
                "alpha": self.alpha,
                "modes": modes,
                "omega": omega,
                "selected_indices": [],
                "reconstructed": x.copy(),
                "scores": [],
                "heart_freq": heart_freq,
                "quality": 0.0,
            }

        scores = sorted(scores, key=lambda item: item["score"], reverse=True)
        best_item = scores[0]
        best_idx = int(best_item["imf_index"])

        reconstructed = modes[best_idx].copy()
        reconstructed = reconstructed - np.mean(reconstructed)
        heart_freq, _, _, _ = self.dominant_heart_freq(reconstructed)

        return {
            "K": self.k,
            "alpha": self.alpha,
            "modes": modes,
            "omega": omega,
            "selected_indices": [best_idx],
            "reconstructed": reconstructed,
            "scores": scores,
            "heart_freq": heart_freq,
            "quality": float(best_item["score"]),
        }

    def _score_imf_fuzzy(self, imf):
        peak_freq, peak_amp, freq, mag = self.dominant_heart_freq(imf)
        heart_mask = (freq >= self.bands.heart[0]) & (freq <= self.bands.heart[1])

        if not np.isfinite(peak_freq) or peak_amp <= 0 or not np.any(heart_mask):
            return None

        heart_mag = mag[heart_mask]
        noise_floor = np.median(heart_mag) + 1e-12

        peak_prominence = float(peak_amp / noise_floor)
        band_ratio = self._band_energy_ratio(freq, mag, self.bands.heart)
        freq_prior = self._freq_prior_membership(peak_freq)
        spectral_concentration = self._spectral_concentration(freq, mag, peak_freq)
        periodicity = self._periodicity_score(imf, peak_freq)
        impulsiveness = self._impulsiveness_score(imf)

        fuzzy = self._fuzzy_heartbeat_score(
            peak_prominence=peak_prominence,
            band_ratio=band_ratio,
            freq_prior=freq_prior,
            spectral_concentration=spectral_concentration,
            periodicity=periodicity,
            impulsiveness=impulsiveness,
        )

        return {
            "score": float(fuzzy["score"]),
            "peak_freq": float(peak_freq),
            "peak_amp": float(peak_amp),
            "peak_prominence": float(peak_prominence),
            "band_ratio": float(band_ratio),
            "freq_prior": float(freq_prior),
            "spectral_concentration": float(spectral_concentration),
            "periodicity": float(periodicity),
            "impulsiveness": float(impulsiveness),
            "membership_peak": float(fuzzy["membership_peak"]),
            "membership_band": float(fuzzy["membership_band"]),
            "membership_concentration": float(fuzzy["membership_concentration"]),
            "membership_periodicity": float(fuzzy["membership_periodicity"]),
            "membership_impulse": float(fuzzy["membership_impulse"]),
        }

    # def _fuzzy_heartbeat_score(
    #     self,
    #     peak_prominence,
    #     band_ratio,
    #     freq_prior,
    #     spectral_concentration,
    #     periodicity,
    #     impulsiveness,
    # ):
    #     membership_peak = self._saturating_membership(peak_prominence, low=2.0, high=8.0)
    #     membership_band = self._saturating_membership(band_ratio, low=0.18, high=0.65)
    #     membership_concentration = self._saturating_membership(spectral_concentration, low=0.20, high=0.58)
    #     membership_periodicity = self._saturating_membership(periodicity, low=0.15, high=0.55)

    #     # 心跳可以有尖脉冲特征，但过强的尖峰更像毛刺，所以这里用梯形隶属度。
    #     membership_impulse = self._trapezoid_membership(
    #         impulsiveness,
    #         left=1.8,
    #         left_top=2.6,
    #         right_top=7.0,
    #         right=11.0,
    #     )

    #     # 模糊规则的加权综合：
    #     # 频率先验和频带能量负责生理合理性；
    #     # 峰突出、谱集中、周期性负责信号质量；
    #     # 冲击性作为辅助项，避免把纯噪声或纯正弦都误选为心跳。
    #     score = (
    #         0.22 * membership_peak
    #         + 0.22 * membership_band
    #         + 0.18 * freq_prior
    #         + 0.16 * membership_concentration
    #         + 0.14 * membership_periodicity
    #         + 0.08 * membership_impulse
    #     )

    #     return {
    #         "score": score,
    #         "membership_peak": membership_peak,
    #         "membership_band": membership_band,
    #         "membership_concentration": membership_concentration,
    #         "membership_periodicity": membership_periodicity,
    #         "membership_impulse": membership_impulse,
    #     }

    def _fuzzy_heartbeat_score(
        self,
        peak_prominence,
        band_ratio,
        freq_prior,
        spectral_concentration,
        periodicity,
        impulsiveness,
    ):
        """
        Mamdani 模糊推理版本。

        为避免 6 输入规则库过大，这里只把 3 个核心变量放入 Mamdani 推理：
        1. peak_prominence：主峰突出度
        2. band_ratio：心跳频带能量占比
        3. periodicity：周期性指标

        freq_prior、spectral_concentration、impulsiveness 保留为诊断量，
        但不参与本版 Mamdani 规则。
        """

        membership_peak = self._saturating_membership(
            peak_prominence,
            low=2.0,
            high=8.0,
        )
        membership_band = self._saturating_membership(
            band_ratio,
            low=0.18,
            high=0.65,
        )
        membership_concentration = self._saturating_membership(
            spectral_concentration,
            low=0.20,
            high=0.58,
        )
        membership_periodicity = self._saturating_membership(
            periodicity,
            low=0.15,
            high=0.55,
        )
        membership_impulse = self._trapezoid_membership(
            impulsiveness,
            left=1.8,
            left_top=2.6,
            right_top=7.0,
            right=11.0,
        )

        mamdani = self._mamdani_heartbeat_score(
            peak_prominence=peak_prominence,
            band_ratio=band_ratio,
            periodicity=periodicity,
        )

        return {
            "score": float(mamdani["score"]),
            "membership_peak": float(membership_peak),
            "membership_band": float(membership_band),
            "membership_concentration": float(membership_concentration),
            "membership_periodicity": float(membership_periodicity),
            "membership_impulse": float(membership_impulse),
            "mamdani_top_rule": mamdani.get("top_rule"),
            "mamdani_input_memberships": mamdani.get("input_memberships"),
        }


    def _mamdani_heartbeat_score(self, peak_prominence, band_ratio, periodicity):
        """
        标准 Mamdani 推理：
        - AND: min
        - 蕴含: min 截断输出隶属度函数
        - 聚合: max
        - 解模糊: centroid 重心法
        """

        if not np.isfinite(peak_prominence):
            peak_prominence = 0.0
        if not np.isfinite(band_ratio):
            band_ratio = 0.0
        if not np.isfinite(periodicity):
            periodicity = 0.0

        P_value = float(np.clip(peak_prominence, 0.0, 12.0))
        B_value = float(np.clip(band_ratio, 0.0, 1.0))
        T_value = float(np.clip(periodicity, 0.0, 1.0))

        P = {
            "L": self._trapmf(P_value, 0.0, 0.0, 2.0, 4.0),
            "M": self._trimf(P_value, 2.5, 5.0, 7.5),
            "H": self._trapmf(P_value, 6.0, 8.0, 12.0, 12.0),
        }

        B = {
            "L": self._trapmf(B_value, 0.0, 0.0, 0.18, 0.32),
            "M": self._trimf(B_value, 0.20, 0.42, 0.65),
            "H": self._trapmf(B_value, 0.50, 0.65, 1.0, 1.0),
        }

        T = {
            "L": self._trapmf(T_value, 0.0, 0.0, 0.12, 0.25),
            "M": self._trimf(T_value, 0.15, 0.40, 0.65),
            "H": self._trapmf(T_value, 0.55, 0.70, 1.0, 1.0),
        }

        q = np.linspace(0.0, 1.0, 501)

        Q = {
            "VL": self._trapmf(q, 0.00, 0.00, 0.10, 0.25),
            "L": self._trimf(q, 0.15, 0.30, 0.48),
            "M": self._trimf(q, 0.38, 0.55, 0.72),
            "H": self._trimf(q, 0.62, 0.78, 0.92),
            "VH": self._trapmf(q, 0.84, 0.93, 1.00, 1.00),
        }

        rule_table = {
            "H": {
                "L": {"L": "VL", "M": "L",  "H": "L"},
                "M": {"L": "L",  "M": "M",  "H": "H"},
                "H": {"L": "M",  "M": "H",  "H": "VH"},
            },
            "M": {
                "L": {"L": "VL", "M": "VL", "H": "L"},
                "M": {"L": "VL", "M": "L",  "H": "M"},
                "H": {"L": "L",  "M": "M",  "H": "H"},
            },
            "L": {
                "L": {"L": "VL", "M": "VL", "H": "VL"},
                "M": {"L": "VL", "M": "VL", "H": "L"},
                "H": {"L": "VL", "M": "L",  "H": "M"},
            },
        }

        aggregated = np.zeros_like(q)
        activated_rules = []

        for t_label, p_block in rule_table.items():
            for p_label, b_block in p_block.items():
                for b_label, q_label in b_block.items():
                    strength = min(T[t_label], P[p_label], B[b_label])
                    if strength <= 0.0:
                        continue

                    clipped_output = np.minimum(strength, Q[q_label])
                    aggregated = np.maximum(aggregated, clipped_output)

                    activated_rules.append({
                        "T": t_label,
                        "P": p_label,
                        "B": b_label,
                        "Q": q_label,
                        "strength": float(strength),
                    })

        denominator = float(np.sum(aggregated))
        if denominator <= 1e-12:
            score = 0.0
        else:
            score = float(np.sum(q * aggregated) / denominator)

        top_rule = None
        if activated_rules:
            top_rule = max(activated_rules, key=lambda item: item["strength"])

        return {
            "score": score,
            "top_rule": top_rule,
            "input_memberships": {
                "P_peak_prominence": P,
                "B_band_ratio": B,
                "T_periodicity": T,
            },
        }


    @staticmethod
    def _trapmf(x, a, b, c, d):
        """
        梯形隶属度函数，支持标量和 numpy 数组。
        允许 a == b 表示左肩函数，允许 c == d 表示右肩函数。
        """
        arr = np.asarray(x, dtype=np.float64)
        scalar_input = arr.ndim == 0
        z = arr.reshape(1) if scalar_input else arr

        y = np.zeros_like(z, dtype=np.float64)

        # 上升沿
        if b > a:
            idx = (z > a) & (z < b)
            y[idx] = (z[idx] - a) / (b - a)

        # 平顶
        idx = (z >= b) & (z <= c)
        y[idx] = 1.0

        # 下降沿
        if d > c:
            idx = (z > c) & (z < d)
            y[idx] = (d - z[idx]) / (d - c)

        y = np.clip(y, 0.0, 1.0)

        if scalar_input:
            return float(y[0])
        return y


    @staticmethod
    def _trimf(x, a, b, c):
        """
        三角形隶属度函数。
        """
        return HeartbeatProcessor._trapmf(x, a, b, b, c)

    def _freq_prior_membership(self, peak_freq):
        return self._trapezoid_membership(
            peak_freq,
            left=self.bands.heart[0],
            left_top=self.heart_peak_prefer_min_hz,
            right_top=self.heart_peak_prefer_max_hz,
            right=self.bands.heart[1],
        )

    def _spectral_concentration(self, freq, mag, peak_freq, half_width=0.12):
        heart_mask = (freq >= self.bands.heart[0]) & (freq <= self.bands.heart[1])
        peak_mask = (freq >= peak_freq - half_width) & (freq <= peak_freq + half_width)

        heart_energy = np.sum(mag[heart_mask] ** 2) + 1e-12
        peak_energy = np.sum(mag[peak_mask] ** 2)

        return float(peak_energy / heart_energy)

    def _periodicity_score(self, x, peak_freq):
        x = np.asarray(x, dtype=np.float64)
        x = x - np.mean(x)

        den = np.dot(x, x)
        if den < 1e-12 or not np.isfinite(peak_freq) or peak_freq <= 0:
            return 0.0

        expected_lag = int(round(self.tools.fs / peak_freq))
        search_radius = max(1, int(round(0.18 * expected_lag)))

        lag_min = max(1, expected_lag - search_radius)
        lag_max = min(len(x) - 2, expected_lag + search_radius)

        if lag_max <= lag_min:
            return 0.0

        best_corr = 0.0
        for lag in range(lag_min, lag_max + 1):
            a = x[:-lag]
            b = x[lag:]
            corr = self.tools.safe_corrcoef(a, b)
            best_corr = max(best_corr, corr)

        return float(np.clip(best_corr, 0.0, 1.0))

    def _impulsiveness_score(self, x):
        x = np.asarray(x, dtype=np.float64)
        x = x - np.mean(x)
        std = np.std(x) + 1e-12
        z = x / std

        # Pearson kurtosis。正弦约 1.5，高斯约 3，尖脉冲通常更大。
        return float(np.mean(z ** 4))

    @staticmethod
    def _band_energy_ratio(freq, mag, band):
        total_mask = (freq >= 0.2) & (freq <= 3.5)
        band_mask = (freq >= band[0]) & (freq <= band[1])

        total = np.sum(mag[total_mask] ** 2) + 1e-12
        band_energy = np.sum(mag[band_mask] ** 2)

        return float(band_energy / total)

    @staticmethod
    def _saturating_membership(x, low, high):
        if not np.isfinite(x):
            return 0.0
        if x <= low:
            return 0.0
        if x >= high:
            return 1.0
        return float((x - low) / (high - low))

    @staticmethod
    def _trapezoid_membership(x, left, left_top, right_top, right):
        if not np.isfinite(x):
            return 0.0
        if x <= left or x >= right:
            return 0.0
        if left_top <= x <= right_top:
            return 1.0
        if left < x < left_top:
            return float((x - left) / (left_top - left))
        return float((right - x) / (right - right_top))
    
    # ============================================================
    # Paper-quality plotting utilities
    # ============================================================

    @staticmethod
    def _cm_to_inch(*values):
        """Convert cm to inch for paper figure size control."""
        return tuple(v / 2.54 for v in values)

    @staticmethod
    def _mf_trap_array(x, a, b, c, d):
        """
        Trapezoidal membership function.
        Supports scalar and numpy array.
        Allows a == b for left-shoulder and c == d for right-shoulder.
        """
        arr = np.asarray(x, dtype=np.float64)
        scalar_input = arr.ndim == 0
        z = arr.reshape(1) if scalar_input else arr

        y = np.zeros_like(z, dtype=np.float64)

        if b > a:
            idx = (z > a) & (z < b)
            y[idx] = (z[idx] - a) / (b - a)

        idx = (z >= b) & (z <= c)
        y[idx] = 1.0

        if d > c:
            idx = (z > c) & (z < d)
            y[idx] = (d - z[idx]) / (d - c)

        y = np.clip(y, 0.0, 1.0)

        if scalar_input:
            return float(y[0])
        return y

    def _mf_tri_array(self, x, a, b, c):
        """Triangular membership function."""
        return self._mf_trap_array(x, a, b, b, c)

    @staticmethod
    def _format_2d_axis(ax, xlabel=None, ylabel=None, xlim=None, ylim=None):
        """Format 2D axes for thesis/paper figures."""
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)


        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            length=3.0,
            width=0.8,
            top=True,
            right=True,
        )

        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    @staticmethod
    def _save_paper_figure(fig, save_path):
        """
        Save figure in paper-friendly format.

        If save_path has no suffix:
            save_path='figs/membership_functions'
            -> figs/membership_functions.pdf
            -> figs/membership_functions.tif

        If save_path has suffix:
            save_path='figs/membership_functions.pdf'
            -> only save this file
        """
        if save_path is None:
            return

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix:
            fig.savefig(
                path,
                bbox_inches="tight",
                pad_inches=0.02,
            )
        else:
            fig.savefig(
                path.with_suffix(".svg"),
                bbox_inches="tight",
                pad_inches=0.02,
            )
            fig.savefig(
                path.with_suffix(".tif"),
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.02,
            )

    def plot_mamdani_membership_functions(
        self,
        save_path=None,
        show=True,
        fig_width_cm=16.0,
        fig_height_cm=10.0,
        ):
            """
            Plot input and output membership functions used by the Mamdani fuzzy inference.

            Paper style:
            - No Chinese text inside figure.
            - No figure title; use caption in the paper.
            - Black-and-white line styles for print compatibility.
            - Save as PDF + 600 dpi TIFF if save_path has no suffix.
            """

            fig, axes = plt.subplots(
                2,
                2,
                figsize=self._cm_to_inch(fig_width_cm, fig_height_cm),
                constrained_layout=True,
            )

            axes = axes.ravel()

            line_styles_3 = {
                "Low": "-",
                "Medium": "--",
                "High": ":",
            }

            line_styles_5 = {
                "Very low": "-",
                "Low": "--",
                "Medium": ":",
                "High": "-.",
                "Very high": (0, (5, 2, 1, 2)),
            }

            # Direct labels placed above the membership curves.
            # Keep the font small for paper figures.
            mf_label_kwargs = dict(
                ha="center",
                va="bottom",
                fontsize=9,
                color="black",
                clip_on=False,
            )

            # --------------------------------------------------------
            # (a) Peak prominence P
            # --------------------------------------------------------
            P = np.linspace(0.0, 12.0, 800)

            P_low = self._mf_trap_array(P, 0.0, 0.0, 2.0, 4.0)
            P_mid = self._mf_tri_array(P, 2.5, 5.0, 7.5)
            P_high = self._mf_trap_array(P, 6.0, 8.0, 12.0, 12.0)

            ax = axes[0]
            ax.plot(P, P_low, color="black", linestyle=line_styles_3["Low"], label="Low")
            ax.plot(P, P_mid, color="black", linestyle=line_styles_3["Medium"], label="Medium")
            ax.plot(P, P_high, color="black", linestyle=line_styles_3["High"], label="High")

            self._format_2d_axis(
                ax,
                xlabel=r"Peak prominence ($P$)",
                ylabel=r"$\mu$",
                xlim=(0.0, 12.0),
                ylim=(-0.03, 1.2)
            )
            ax.set_yticks([0.0, 0.5, 1.0])
            ax.text(-0.1, 1.0, "(a)", transform=ax.transAxes)
            ax.text(1.0, 1.05, "L", **mf_label_kwargs)
            ax.text(5.0, 1.05, "M", **mf_label_kwargs)
            ax.text(10.0, 1.05, "H", **mf_label_kwargs)
            # ax.legend(frameon=False, loc="upper right", handlelength=2.5)

            # --------------------------------------------------------
            # (b) Band energy ratio B
            # --------------------------------------------------------
            B = np.linspace(0.0, 1.0, 800)

            B_low = self._mf_trap_array(B, 0.0, 0.0, 0.18, 0.32)
            B_mid = self._mf_tri_array(B, 0.20, 0.42, 0.65)
            B_high = self._mf_trap_array(B, 0.50, 0.65, 1.0, 1.0)

            ax = axes[1]
            ax.plot(B, B_low, color="black", linestyle=line_styles_3["Low"], label="Low")
            ax.plot(B, B_mid, color="black", linestyle=line_styles_3["Medium"], label="Medium")
            ax.plot(B, B_high, color="black", linestyle=line_styles_3["High"], label="High")

            self._format_2d_axis(
                ax,
                xlabel=r"Band energy ratio ($E$)",
                ylabel=r"$\mu$",
                xlim=(0.0, 1.0),
                ylim=(-0.03, 1.2)
            )
            ax.text(-0.1, 1.0, "(b)", transform=ax.transAxes)
            ax.text(0.09, 1.05, "L", **mf_label_kwargs)
            ax.text(0.42, 1.05, "M", **mf_label_kwargs)
            ax.text(0.82, 1.05, "H", **mf_label_kwargs)
            ax.set_yticks([0.0, 0.5, 1.0])
            # ax.legend(frameon=False, loc="upper right", handlelength=2.5)

            # --------------------------------------------------------
            # (c) Periodicity T
            # --------------------------------------------------------
            T = np.linspace(0.0, 1.0, 800)

            T_low = self._mf_trap_array(T, 0.0, 0.0, 0.12, 0.25)
            T_mid = self._mf_tri_array(T, 0.15, 0.40, 0.65)
            T_high = self._mf_trap_array(T, 0.55, 0.70, 1.0, 1.0)

            ax = axes[2]
            ax.plot(T, T_low, color="black", linestyle=line_styles_3["Low"], label="Low")
            ax.plot(T, T_mid, color="black", linestyle=line_styles_3["Medium"], label="Medium")
            ax.plot(T, T_high, color="black", linestyle=line_styles_3["High"], label="High")

            self._format_2d_axis(
                ax,
                xlabel=r"Periodicity ($T$)",
                ylabel=r"$\mu$",
                xlim=(0.0, 1.0),
                ylim=(-0.03, 1.2)
            )
            ax.text(-0.1, 1.0, "(c)", transform=ax.transAxes)
            ax.text(0.08, 1.05, "L", **mf_label_kwargs)
            ax.text(0.40, 1.05, "M", **mf_label_kwargs)
            ax.text(0.82, 1.05, "H", **mf_label_kwargs)
            ax.set_yticks([0.0, 0.5, 1.0])
            # ax.legend(frameon=False, loc="upper right", handlelength=2.5)

            # --------------------------------------------------------
            # (d) Output score Q
            # --------------------------------------------------------
            Q = np.linspace(0.0, 1.0, 800)

            Q_vl = self._mf_trap_array(Q, 0.00, 0.00, 0.10, 0.25)
            Q_l = self._mf_tri_array(Q, 0.15, 0.30, 0.48)
            Q_m = self._mf_tri_array(Q, 0.38, 0.55, 0.72)
            Q_h = self._mf_tri_array(Q, 0.62, 0.78, 0.92)
            Q_vh = self._mf_trap_array(Q, 0.84, 0.93, 1.00, 1.00)

            ax = axes[3]
            ax.plot(Q, Q_vl, color="black", linestyle=line_styles_5["Very low"], label="Very low")
            ax.plot(Q, Q_l, color="black", linestyle=line_styles_5["Low"], label="Low")
            ax.plot(Q, Q_m, color="black", linestyle=line_styles_5["Medium"], label="Medium")
            ax.plot(Q, Q_h, color="black", linestyle=line_styles_5["High"], label="High")
            ax.plot(Q, Q_vh, color="black", linestyle=line_styles_5["Very high"], label="Very high")

            self._format_2d_axis(
                ax,
                xlabel=r"Heartbeat IMF score($Q$)",
                ylabel=r"$\mu$",
                xlim=(0.0, 1.0),
                ylim=(-0.03, 1.2)
            )
            ax.text(-0.1, 1.0, "(d)", transform=ax.transAxes)
            ax.text(0.08, 1.05, "VL", **mf_label_kwargs)
            ax.text(0.30, 1.05, "L", **mf_label_kwargs)
            ax.text(0.55, 1.05, "M", **mf_label_kwargs)
            ax.text(0.78, 1.05, "H", **mf_label_kwargs)
            ax.text(0.95, 1.05, "VH", **mf_label_kwargs)
            ax.set_yticks([0.0, 0.5, 1.0])
            # ax.legend(frameon=False, loc="upper center", ncol=2, handlelength=2.5)

            self._save_paper_figure(fig, save_path)

            if show:
                plt.show()
            else:
                plt.close(fig)

            return fig, axes
    
    # def plot_vmd_components_one_window(
    #     self,
    #     heart_input,
    #     vmd_result,
    #     start_sec=0.0,
    #     duration_sec=8.0,
    #     save_path=None,
    #     show=True,
    #     normalize=False,
    #     include_input=True,
    #     fig_width_cm=16.0,
    # ):
    #     """
    #     Plot VMD input and IMF components in one figure window.

    #     Parameters
    #     ----------
    #     heart_input : array-like
    #         Heart-band signal before VMD.
    #     vmd_result : dict
    #         Output of self.extract() or self._run_vmd_once().
    #         Must contain vmd_result["modes"].
    #     start_sec : float
    #         Start time of the displayed window.
    #     duration_sec : float or None
    #         Display duration in seconds. If None, plot until the end.
    #     save_path : str or Path or None
    #         If no suffix is given, save both PDF and TIFF.
    #     show : bool
    #         True: show the figure.
    #         False: save and close without showing.
    #     normalize : bool
    #         True: normalize each row by its maximum absolute amplitude.
    #         This improves visual comparison but changes the displayed scale.
    #     include_input : bool
    #         True: include the VMD input signal as the first row.
    #     fig_width_cm : float
    #         Figure width in cm.
    #     """

    #     x = np.asarray(heart_input, dtype=np.float64).reshape(-1)

    #     if "modes" not in vmd_result:
    #         raise KeyError('vmd_result must contain key "modes".')

    #     modes = np.asarray(vmd_result["modes"], dtype=np.float64)

    #     if modes.ndim != 2:
    #         raise ValueError(
    #             'vmd_result["modes"] must be a 2D array with shape (K, N).'
    #         )

    #     fs = float(self.tools.fs)
    #     n = min(len(x), modes.shape[1])

    #     if n <= 1:
    #         raise ValueError("Signal length is too short for plotting.")

    #     start_idx = int(round(float(start_sec) * fs))
    #     start_idx = max(0, min(start_idx, n - 1))

    #     if duration_sec is None:
    #         end_idx = n
    #     else:
    #         win_len = int(round(float(duration_sec) * fs))
    #         win_len = max(2, win_len)
    #         end_idx = min(n, start_idx + win_len)

    #     if end_idx <= start_idx + 1:
    #         raise ValueError("Invalid plotting window. Check start_sec and duration_sec.")

    #     # Relative time within the plotted window.
    #     t = (np.arange(start_idx, end_idx) - start_idx) / fs

    #     selected_indices = set(
    #         int(idx) for idx in vmd_result.get("selected_indices", [])
    #     )

    #     series = []

    #     if include_input:
    #         series.append(("Input", x[start_idx:end_idx], False))

    #     for i in range(modes.shape[0]):
    #         label = f"IMF {i + 1}"
    #         is_selected = i in selected_indices

    #         if is_selected:
    #             label += r"$^\ast$"

    #         series.append((label, modes[i, start_idx:end_idx], is_selected))

    #     n_rows = len(series)

    #     # Height is adapted to number of components but controlled for paper layout.
    #     fig_height_cm = max(6.0, 1.30 * n_rows + 1.0)

    #     fig, axes = plt.subplots(
    #         n_rows,
    #         1,
    #         sharex=True,
    #         figsize=self._cm_to_inch(fig_width_cm, fig_height_cm),
    #     )

    #     if n_rows == 1:
    #         axes = [axes]

    #     for ax, (label, y, is_selected) in zip(axes, series):
    #         y_plot = np.asarray(y, dtype=np.float64)

    #         if normalize:
    #             scale = np.max(np.abs(y_plot)) + 1e-12
    #             y_plot = y_plot / scale

    #         line_width = 1.15 if is_selected else 0.85

    #         ax.plot(
    #             t,
    #             y_plot,
    #             color="black",
    #             linewidth=line_width,
    #         )

    #         ax.text(
    #             0.012,
    #             0.78,
    #             label,
    #             transform=ax.transAxes,
    #             ha="left",
    #             va="center",
    #             fontsize=10,
    #         )

    #         ax.tick_params(
    #             axis="both",
    #             which="both",
    #             direction="in",
    #             length=3.0,
    #             width=0.8,
    #             top=True,
    #             right=True,
    #         )

    #         ax.yaxis.set_major_locator(MaxNLocator(3))

    #         for spine in ax.spines.values():
    #             spine.set_linewidth(0.8)

    #     axes[-1].set_xlabel("Time (s)")

    #     if normalize:
    #         y_label = "Normalized amplitude"
    #     else:
    #         y_label = "Amplitude (rad)"

    #     fig.text(
    #         0.018,
    #         0.50,
    #         y_label,
    #         rotation="vertical",
    #         va="center",
    #         ha="center",
    #     )

    #     axes[-1].set_xlim(t[0], t[-1])

    #     fig.subplots_adjust(
    #         left=0.10,
    #         right=0.985,
    #         bottom=0.08,
    #         top=0.985,
    #         hspace=0.16,
    #     )

    #     self._save_paper_figure(fig, save_path)

    #     if show:
    #         plt.show()
    #     else:
    #         plt.close(fig)

    #     return fig, axes


    def plot_vmd_imfs_for_window(
        self,
        vmd_result,
        window_start_sec,
        window_end_sec=None,
        save_path=None,
        show=False,
        normalize=False,
        fig_width_cm=16.0,
        color="#0072B2",
    ):
        """
        Plot all VMD IMF components of one sliding window in a single figure.

        This function uses the absolute time axis of the original signal,
        e.g. 0-10 s, 10-20 s, 20-30 s when window_sec=10.
        It does not mark the selected IMF and follows paper-style subplot layout.
        """

        if "modes" not in vmd_result:
            raise KeyError('vmd_result must contain key "modes".')

        modes = np.asarray(vmd_result["modes"], dtype=np.float64)

        if modes.ndim != 2:
            raise ValueError('vmd_result["modes"] must be a 2D array with shape (K, N).')

        k, n = modes.shape
        if k <= 0 or n <= 1:
            raise ValueError("VMD modes are too short for plotting.")

        fs = float(self.tools.fs)
        window_start_sec = float(window_start_sec)

        if window_end_sec is None:
            window_end_sec = window_start_sec + n / fs
        else:
            window_end_sec = float(window_end_sec)

        t = window_start_sec + np.arange(n, dtype=np.float64) / fs

        n_rows = int(k)
        fig_height_cm = max(5.8, 1.18 * n_rows + 0.9)

        fig, axes = plt.subplots(
            n_rows,
            1,
            sharex=True,
            figsize=self._cm_to_inch(fig_width_cm, fig_height_cm),
        )

        if n_rows == 1:
            axes = [axes]

        labels = [f"({chr(97 + i)}) IMF {i + 1}" for i in range(n_rows)]

        for i, ax in enumerate(axes):
            y_plot = modes[i].copy()

            if normalize:
                scale = np.max(np.abs(y_plot)) + 1e-12
                y_plot = y_plot / scale

            ax.plot(
                t,
                y_plot,
                color=color,
                linewidth=1.0,
            )

            # ax.text(
            #     0.012,
            #     0.84,
            #     labels[i],
            #     transform=ax.transAxes,
            #     ha="left",
            #     va="center",
            #     fontsize=10,
            # )

            ax.tick_params(
                axis="both",
                which="both",
                direction="in",
                length=3.0,
                width=0.8,
                top=True,
                right=True,
            )

            ax.yaxis.set_major_locator(MaxNLocator(3))
            ax.grid(False)

            for spine in ax.spines.values():
                spine.set_linewidth(0.8)

        axes[-1].set_xlabel("Time (s)")
        axes[-1].set_xlim(window_start_sec, window_end_sec) 

        y_label = "Normalized amplitude" if normalize else "Amplitude (rad)"
        fig.text(
            0.020,
            0.50,
            y_label,
            rotation="vertical",
            va="center",
            ha="center",
        )

        fig.subplots_adjust(
            left=0.11,
            right=0.985,
            bottom=0.08,
            top=0.99,
            hspace=0.14,
        )

        self._save_paper_figure(fig, save_path)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axes
