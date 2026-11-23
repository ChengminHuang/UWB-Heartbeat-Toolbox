# 参数说明（Parameters）

用户主要通过 `configs/default.json` 调整参数，由 `examples/run_and_export.py` 读取配置并调用 `BreatheAndHeartSlidingWindowed`。修改 JSON 参数不需要修改 `src/` 中的算法代码。

## 1. 如何设置和运行

1. 在 `examples/run_and_export.py` 中设置 `input_file`（测量 JSON）、`empty_file`（空场 JSON，可设为 `None`）和 `output_dir`（结果保存目录）。
2. 在 `configs/default.json` 中修改需要调整的算法参数。

## 2. 可调参数一览

| 参数名 | 原配置值 | 作用与说明 |
| --- | --- | --- |
| `range_min_m` | `0.3` | 目标搜索距离下限，单位 m。 |
| `range_max_m` | `7.65` | 目标搜索距离上限，单位 m。 |
| `window_sec` | `20.0` | 滑动窗口长度，单位 s；决定每次分析使用多长时间的数据。 |
| `hop_sec` | `1.0` | 滑动步长，单位 s；决定相邻窗口起点的时间间隔。 |
| `plot_every` | `1` | 每隔多少个窗口更新一次内部动态图；可改为 `10` 减少绘图开销，不改变信号处理计算。必须为正整数。 |
| `detection_threshold` | `2000.0` | 距离门动态响应评分阈值，用于筛选候选目标距离门。 |
| `complex_var_threshold` | `null` | 窗口复数方差评分门限；`null` 表示采用 `detection_threshold` 的值。低于门限的窗口直接输出无效结果。 |
| `resp_null_amplitude` | `0.1` | 呼吸能量判定的参考正弦幅值；用于计算呼吸信号的最低参考能量。 |
| `heart_null_amplitude` | `0.01` | 心跳能量判定的参考正弦幅值；用于计算心跳信号的最低参考能量。 |
| `vmd_k` | `5` | VMD 分解的模态数量（IMF 数）。 |
| `vmd_alpha` | `1000.0` | VMD 的带宽惩罚参数 `alpha`。 |

### 关键参数的调整思路

- **`window_sec`**：窗口较长通常有利于频率分辨，但时间响应较慢；窗口太短可能无法稳定估计低频呼吸。程序还要求每窗至少有 16 个采样点，且不能超过输入数据总长度。
- **`hop_sec`**：步长较小，会产生更多重叠窗口及更密的估计结果，但计算量和耗时也会增加；必须大于零。
- **`detection_threshold`**：过高可能漏检，过低可能把干扰当作目标。其数值与数据幅值有关，不是不同雷达数据通用的固定门限。
- **`complex_var_threshold`**：控制窗口是否进入后续生命体征分析；保持 `null` 时与距离门评分阈值相同。
- **`resp_null_amplitude` / `heart_null_amplitude`**：这两个值不是直接用作波形幅值比较，而是先换算为参考正弦的能量阈值。增大幅值会提高对应能量门限，可能使更多低能量窗口被判为无效。
- **`vmd_k` / `vmd_alpha`**：分别控制分解模态数与带宽约束，改变它们可能影响 IMF 分解和最终心率结果。建议一次只改变一个参数，并记录实验结果。

## 3. 配置文件示例

```json
{
  "range_min_m": 0.3,
  "range_max_m": 7.65,
  "window_sec": 20.0,
  "hop_sec": 1.0,
  "plot_every": 1,
  "detection_threshold": 2000.0,
  "complex_var_threshold": null,
  "resp_null_amplitude": 0.1,
  "heart_null_amplitude": 0.01,
  "vmd_k": 5,
  "vmd_alpha": 1000.0
}
```