
import numpy as np
from scipy import signal

class SignalTools:
    """通用信号处理工具"""

    def __init__(self, fs):
        self.fs = float(fs)

    def fft_amp(self, x, window_type="blackman"):
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        if n < 4:
            raise ValueError("signal is too short for FFT")

        if window_type == "hann":
            window = np.hanning(n)
        elif window_type == "hamming":
            window = np.hamming(n)
        elif window_type == "blackman":
            window = np.blackman(n)
        elif window_type in ("rectangle", "none"):
            window = np.ones(n)
        else:
            raise ValueError("unsupported window type")

        scale = np.sum(window)
        if abs(scale) < 1e-12:
            window = np.ones(n)
            scale = float(n)

        # 去均值可以减小 0 Hz 附近能量对峰值判断的干扰。
        spectrum = np.fft.fft((x - np.mean(x)) * window) / scale * 2.0
        freqs = np.fft.fftfreq(n, d=1.0 / self.fs)
        positive_freqs = freqs[: n // 2]
        positive_fft = np.abs(spectrum[: n // 2])
        positive_fft[0] /= 2.0
        if n % 2 == 0 and len(positive_fft) > 1:
            positive_fft[-1] /= 2.0
        return positive_freqs, positive_fft

    def bandpass(self, x, low, high, order=3):
        x = np.asarray(x, dtype=np.float64)
        nyquist = 0.5 * self.fs
        low_n = max(low / nyquist, 1e-6)
        high_n = min(high / nyquist, 0.999)
        if high_n <= low_n:
            raise ValueError(f"invalid bandpass: {low}-{high} Hz at fs={self.fs:.3f} Hz")

        b, a = signal.butter(order, [low_n, high_n], btype="band")
        padlen = 3 * max(len(a), len(b))
        if len(x) <= padlen:
            # 窗口极短时 filtfilt 会失败，退化成单向滤波。
            return signal.lfilter(b, a, x)
        return signal.filtfilt(b, a, x)

    def detrend_phase(self, phase_unwrap, use_savgol=False, sg_window_default=801, sg_polyorder=3):
        if use_savgol:
            win = self._safe_savgol_window(len(phase_unwrap), sg_window_default, sg_polyorder)
            if win is not None:
                baseline = signal.savgol_filter(
                    phase_unwrap,
                    window_length=win,
                    polyorder=min(sg_polyorder, win - 1),
                )
                return phase_unwrap - baseline
        return signal.detrend(phase_unwrap, type="linear")

    def safe_corrcoef(self, a, b):
        a = np.asarray(a, dtype=np.float64) - np.mean(a)
        b = np.asarray(b, dtype=np.float64) - np.mean(b)
        den = np.linalg.norm(a) * np.linalg.norm(b)
        if den < 1e-12:
            return 0.0
        return float(np.dot(a, b) / den)

    def match_sign(self, prev_window, curr_window, overlap_len, flip_threshold=-0.50, keep_threshold=0.50):
        """利用重叠区域相关系数判断相邻窗口是否需要整体翻转符号。"""

        if prev_window is None or overlap_len <= 0:
            return curr_window, np.nan, "init"

        corr = self.safe_corrcoef(prev_window[-overlap_len:], curr_window[:overlap_len])
        if corr <= flip_threshold:
            return -curr_window, corr, "flip"
        if corr >= keep_threshold:
            return curr_window, corr, "keep"
        return curr_window, corr, "ambiguous"

    def dominant_freq(self, x, band):
        freq, mag = self.fft_amp(x)
        mask = (freq >= band[0]) & (freq <= band[1])
        if not np.any(mask):
            return np.nan, 0.0, freq, mag

        local_freq = freq[mask]
        local_mag = mag[mask]
        idx = int(np.argmax(local_mag))
        return float(local_freq[idx]), float(local_mag[idx]), freq, mag

    @staticmethod
    def normalize_ola(accum, weight):
        return np.divide(
            accum,
            np.maximum(weight, 1e-12),
            out=np.zeros_like(accum),
            where=weight > 0,
        )

    @staticmethod
    def _safe_savgol_window(n, wanted, polyorder):
        if n < 7:
            return None
        win = min(wanted, n - 1 if n % 2 == 0 else n)
        if win % 2 == 0:
            win -= 1
        if win < polyorder + 2:
            return None
        return win
