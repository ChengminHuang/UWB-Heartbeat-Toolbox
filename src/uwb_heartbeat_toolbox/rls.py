

import numpy as np

class RespirationProcessor:
    """呼吸处理类。
    1. 从主门总相位中滤出呼吸波形；
    2. 估计呼吸频率；
    3. 用 RLS 递推拟合并扣除呼吸及其谐波。
    """

    def __init__(
        self,
        tools,
        bands,
        max_harmonics=5,
        use_rls=True,
        rls_forgetting=0.995,
        rls_init_cov=1000.0,
        rls_freq_reset_hz=0.06,
    ):
        self.tools = tools
        self.bands = bands
        self.max_harmonics = int(max_harmonics)

        self.use_rls = bool(use_rls)
        self.rls_forgetting = float(rls_forgetting)
        self.rls_init_cov = float(rls_init_cov)
        self.rls_freq_reset_hz = float(rls_freq_reset_hz)

        self._rls_theta = None
        self._rls_p = None
        self._rls_last_used_freqs = None

    def extract(self, total_window):
        """从输入的 total_window中提取呼吸波形和呼吸主频。"""
        resp_wave = self.tools.bandpass(total_window, self.bands.resp[0], self.bands.resp[1], order=3)
        resp_freq, resp_amp, _, _ = self.tools.dominant_freq(resp_wave, self.bands.resp)
        return resp_wave, resp_freq, resp_amp

    def fit_and_remove_harmonics(self, total_window, resp_freq, start_index=0, update_from_index=None):
        """递推最小二乘拟合呼吸及其谐波，然后从总相位中扣除。

        start_index 是当前窗口在原始序列中的起点。
        update_from_index 是本次需要递推更新的原始序列起点。
        后续滑窗只更新新进入的 hop 段，从而减少重复计算。
        """

        x = np.asarray(total_window, dtype=np.float64)
        used_freqs = self._get_used_harmonics(resp_freq) # 确定参与拟合的频率成分
        a_matrix = self._build_harmonic_matrix(len(x), used_freqs) # 构建设计矩阵

        if not self.use_rls:
            coef, _, _, _ = np.linalg.lstsq(a_matrix, x, rcond=None)
            resp_fit = a_matrix @ coef
            residual = x - resp_fit
            return resp_fit, residual, used_freqs

        n_params = a_matrix.shape[1]
        need_reset = self._rls_need_reset(n_params, used_freqs)

        if need_reset:
            self._reset_rls(n_params)
            update_start = 0
        else:
            if update_from_index is None:
                update_start = 0
            else:
                update_start = int(update_from_index - start_index)
                update_start = int(np.clip(update_start, 0, len(x) - 1))

        for phi, y in zip(a_matrix[update_start:], x[update_start:]):
            self._rls_step(phi, y)

        self._rls_last_used_freqs = np.asarray(used_freqs, dtype=np.float64)

        resp_fit = a_matrix @ self._rls_theta
        residual = x - resp_fit
        return resp_fit, residual, used_freqs

    def _get_used_harmonics(self, resp_freq):
        used_freqs = []

        if np.isfinite(resp_freq) and resp_freq > 0:
            for k in range(1, self.max_harmonics + 1):
                fk = k * resp_freq
                if fk > self.bands.heart[1] + 0.2 or fk >= 0.45 * self.tools.fs:
                    break
                used_freqs.append(float(fk))

        return used_freqs

    def _build_harmonic_matrix(self, n, used_freqs):
        t = np.arange(n, dtype=np.float64) / self.tools.fs
        cols = [np.ones_like(t), t - np.mean(t)]

        for fk in used_freqs:
            cols.append(np.sin(2.0 * np.pi * fk * t))
            cols.append(np.cos(2.0 * np.pi * fk * t))

        return np.column_stack(cols)

    def _rls_need_reset(self, n_params, used_freqs):
        if self._rls_theta is None or self._rls_p is None:
            return True

        if len(self._rls_theta) != n_params:
            return True

        if self._rls_last_used_freqs is None:
            return True

        last_freqs = np.asarray(self._rls_last_used_freqs, dtype=np.float64)
        curr_freqs = np.asarray(used_freqs, dtype=np.float64)

        if len(last_freqs) != len(curr_freqs):
            return True

        if len(curr_freqs) > 0:
            max_shift = np.max(np.abs(curr_freqs - last_freqs))
            if max_shift > self.rls_freq_reset_hz:
                return True

        return False

    def _reset_rls(self, n_params):
        self._rls_theta = np.zeros(n_params, dtype=np.float64)
        self._rls_p = np.eye(n_params, dtype=np.float64) * self.rls_init_cov

    def _rls_step(self, phi, y):
        phi = np.asarray(phi, dtype=np.float64)
        y = float(y)

        p_phi = self._rls_p @ phi
        denom = self.rls_forgetting + float(phi @ p_phi)

        if abs(denom) < 1e-12:
            return

        gain = p_phi / denom
        error = y - float(phi @ self._rls_theta)

        self._rls_theta = self._rls_theta + gain * error
        self._rls_p = (self._rls_p - np.outer(gain, p_phi)) / self.rls_forgetting
        self._rls_p = 0.5 * (self._rls_p + self._rls_p.T)
