

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    # 字号：坐标轴小四约 12 pt，图例可略小
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,

    # 线条和边框
    "axes.linewidth": 1.0,
    "lines.linewidth": 1.0,

    # 保存质量
    "savefig.dpi": 600,
    "figure.dpi": 150,
})

class CumulativePlotter:
    """滑动窗口和最终 OLA 结果的论文风格绘图类。"""

    def __init__(self, axis_time, tools, bands, heartbeat_processor):
        self.axis_time = axis_time
        self.tools = tools
        self.bands = bands
        self.heartbeat = heartbeat_processor
        self.fig = None
        self.ax_resp_time = None
        self.ax_resp_fft = None
        self.ax_heart_time = None
        self.ax_heart_fft = None

        # Color-blind friendly palette. Keep red only for peak markers.
        self.signal_color = "#0072B2"
        self.peak_color = "#D62728"

    @staticmethod
    def _cm_to_inch(*values):
        return tuple(v / 2.54 for v in values)

    @staticmethod
    def _format_axis(ax, xlabel=None, ylabel=None, xlim=None, ylim=None):
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.grid(False)
        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            top=True,
            right=True,
            length=3.0,
            width=0.8,
        )
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    @staticmethod
    def _save_figure(fig, save_path):
        """保存论文图片；无后缀时同时保存 SVG 和 600 dpi TIFF。"""
        if save_path is None:
            return

        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
        else:
            fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
            fig.savefig(path.with_suffix(".tif"), dpi=600, bbox_inches="tight", pad_inches=0.02)

    def setup_dynamic(self, fig_width_cm=16.0, fig_height_cm=10.0):
        """初始化动态窗口图：始终为当前窗口的 2 x 2 子图。"""
        plt.ion()
        self.fig, axes = plt.subplots(
            2,
            2,
            figsize=self._cm_to_inch(fig_width_cm, fig_height_cm),
            constrained_layout=True,
        )
        self.ax_resp_time = axes[0, 0]
        self.ax_resp_fft = axes[0, 1]
        self.ax_heart_time = axes[1, 0]
        self.ax_heart_fft = axes[1, 1]

    def update_dynamic(
        self,
        win_idx,
        window_start_sec,
        resp_window,
        heart_window,
        resp_freq,
        heart_freq,
    ):
        """只绘制当前滑动窗口的呼吸、心跳时域和频域四子图。"""
        resp_window = np.asarray(resp_window, dtype=np.float64)
        heart_window = np.asarray(heart_window, dtype=np.float64)
        n = min(len(resp_window), len(heart_window))
        if n <= 1:
            return

        resp_seg = resp_window[:n]
        heart_seg = heart_window[:n]
        t = float(window_start_sec) + np.arange(n, dtype=np.float64) / self.tools.fs
        window_end_sec = t[-1]

        self.ax_resp_time.cla()
        self.ax_resp_time.plot(t, resp_seg, color=self.signal_color, linewidth=1.0)
        self._format_axis(
            self.ax_resp_time,
            xlabel="Time (s)",
            ylabel="Amplitude (rad)",
            xlim=(t[0], window_end_sec),
        )
        self.ax_resp_time.text(-0.12, 1.04, "(a)", transform=self.ax_resp_time.transAxes)
        self.ax_resp_time.text(
            0.98,
            0.92,
            f"Window {win_idx:03d}",
            transform=self.ax_resp_time.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )

        self.ax_resp_fft.cla()
        self.plot_spectrum_with_peak(
            self.ax_resp_fft,
            resp_seg,
            self.bands.resp,
            self.signal_color,
            None,
            (0.0, 1.2),
            peak_freq=resp_freq,
        )
        self.ax_resp_fft.text(-0.12, 1.04, "(b)", transform=self.ax_resp_fft.transAxes)

        self.ax_heart_time.cla()
        self.ax_heart_time.plot(t, heart_seg, color=self.signal_color, linewidth=1.0)
        self._format_axis(
            self.ax_heart_time,
            xlabel="Time (s)",
            ylabel="Amplitude (rad)",
            xlim=(t[0], window_end_sec),
        )
        self.ax_heart_time.text(-0.12, 1.04, "(c)", transform=self.ax_heart_time.transAxes)

        self.ax_heart_fft.cla()
        self.plot_spectrum_with_peak(
            self.ax_heart_fft,
            heart_seg,
            self.bands.heart,
            self.signal_color,
            None,
            (0.0, 3.0),
            peak_freq=heart_freq,
        )
        self.ax_heart_fft.text(-0.12, 1.04, "(d)", transform=self.ax_heart_fft.transAxes)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def plot_range_gate_curve(
        self,
        centers,
        selected_bins,
        raw_bins,
        range_axis_m,
        n_0,
        save_path=None,
        show=True,
    ):
        selected_ranges = range_axis_m[np.asarray(selected_bins, dtype=int)]
        raw_bins = np.asarray(raw_bins, dtype=int)
        raw_ranges = np.where(raw_bins > n_0, range_axis_m[raw_bins], np.nan)

        fig, ax = plt.subplots(figsize=self._cm_to_inch(16.0, 6.0), constrained_layout=True)
        ax.plot(
            centers,
            selected_ranges,
            marker="o",
            markersize=3,
            linewidth=1.1,
            color=self.signal_color,
            label="Selected gate",
        )
        ax.plot(
            centers,
            raw_ranges,
            marker=".",
            markersize=3,
            linewidth=0.8,
            color="black",
            alpha=0.45,
            label="Raw maximum gate",
        )
        self._format_axis(
            ax,
            xlabel="Window center time (s)",
            ylabel="Distance (m)",
        )
        ax.legend(frameon=False, loc="best", handlelength=2.2)
        self._save_figure(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, ax

    def plot_final_ola_results(
        self,
        resp_signal,
        heart_signal,
        save_path=None,
        show=True,
    ):
        """绘制最终 OLA 累积后的呼吸和心跳时域/频域结果。"""
        resp_signal = np.asarray(resp_signal, dtype=np.float64)
        heart_signal = np.asarray(heart_signal, dtype=np.float64)
        n = min(len(resp_signal), len(heart_signal), len(self.axis_time))
        if n <= 1:
            raise ValueError("final OLA signal is too short for plotting")

        t = np.asarray(self.axis_time[:n], dtype=np.float64)
        resp_seg = resp_signal[:n]
        heart_seg = heart_signal[:n]
        valid = np.isfinite(resp_seg) & np.isfinite(heart_seg)
        if not np.any(valid):
            raise ValueError("final OLA signal contains no finite samples")

        fig, axes = plt.subplots(
            2,
            2,
            figsize=self._cm_to_inch(16.0, 10.0),
            constrained_layout=True,
        )

        axes[0, 0].plot(t, resp_seg, color=self.signal_color, linewidth=1.0)
        self._format_axis(axes[0, 0], xlabel="Time (s)", ylabel="Amplitude (rad)", xlim=(t[0], t[-1]))
        axes[0, 0].text(-0.12, 1.04, "(a)", transform=axes[0, 0].transAxes)

        self.plot_spectrum_with_peak(
            axes[0, 1],
            resp_seg,
            self.bands.resp,
            self.signal_color,
            None,
            (0.0, 1.2),
        )
        axes[0, 1].text(-0.12, 1.04, "(b)", transform=axes[0, 1].transAxes)

        axes[1, 0].plot(t, heart_seg, color=self.signal_color, linewidth=1.0)
        self._format_axis(axes[1, 0], xlabel="Time (s)", ylabel="Amplitude (rad)", xlim=(t[0], t[-1]))
        axes[1, 0].text(-0.12, 1.04, "(c)", transform=axes[1, 0].transAxes)

        self.plot_spectrum_with_peak(
            axes[1, 1],
            heart_seg,
            self.bands.heart,
            self.signal_color,
            None,
            (0.0, 3.0),
        )
        axes[1, 1].text(-0.12, 1.04, "(d)", transform=axes[1, 1].transAxes)

        self._save_figure(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_window_rate_curve(
        self,
        centers,
        resp_freqs,
        heart_freqs,
        save_path=None,
        show=True,
    ):
        """绘制逐窗口呼吸率和心率曲线，单位 BPM。"""
        centers = np.asarray(centers, dtype=np.float64)
        resp_bpm = np.asarray(resp_freqs, dtype=np.float64) * 60.0
        heart_bpm = np.asarray(heart_freqs, dtype=np.float64) * 60.0

        fig, axes = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=self._cm_to_inch(16.0, 8.0),
            constrained_layout=True,
        )

        axes[0].plot(
            centers,
            resp_bpm,
            marker="o",
            markersize=3,
            linewidth=1.0,
            color=self.signal_color,
        )
        self._format_axis(axes[0], ylabel="Respiration rate (BPM)")
        axes[0].text(-0.08, 1.04, "(a)", transform=axes[0].transAxes)

        axes[1].plot(
            centers,
            heart_bpm,
            marker="s",
            markersize=3,
            linewidth=1.0,
            color=self.signal_color,
        )
        self._format_axis(axes[1], xlabel="Window center time (s)", ylabel="Heart rate (BPM)")
        axes[1].text(-0.08, 1.04, "(b)", transform=axes[1].transAxes)

        if len(centers) > 1:
            axes[1].set_xlim(float(np.nanmin(centers)), float(np.nanmax(centers)))

        self._save_figure(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig, axes

    def plot_spectrum_with_peak(self, ax, x, band, color, title, xlim, peak_freq=None):
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 4 or np.std(x) < 1e-12:
            ax.plot([], [])
            self._format_axis(ax, xlabel="Frequency (Hz)", ylabel="Magnitude", xlim=xlim)
            return

        freq, mag = self.tools.fft_amp(x)
        if peak_freq is None:
            peak_freq, peak_amp, _, _ = self.tools.dominant_freq(x, band)
        elif np.isfinite(peak_freq):
            idx = int(np.argmin(np.abs(freq - peak_freq)))
            peak_amp = float(mag[idx])
        else:
            peak_amp = 0.0

        ax.plot(freq, mag, color=color, linewidth=1.0)
        if np.isfinite(peak_freq):
            ax.axvline(peak_freq, color=self.peak_color, linestyle="--", linewidth=0.9)
            y_text = peak_amp * 0.82 if peak_amp > 0 else 0.0
            x_text = min(peak_freq + 0.08, xlim[1] - 0.02)
            ax.annotate(
                f"{peak_freq:.3f} Hz\n{peak_freq * 60.0:.1f} BPM",
                xy=(peak_freq, peak_amp),
                xytext=(x_text, y_text),
                arrowprops=dict(arrowstyle="->", color=self.peak_color, lw=0.8, shrinkA=0, shrinkB=0),
                color=self.peak_color,
                fontsize=10,
                ha="left",
                va="top",
            )
        self._format_axis(ax, xlabel="Frequency (Hz)", ylabel="Magnitude", xlim=xlim)
        if title:
            ax.set_title(title)
