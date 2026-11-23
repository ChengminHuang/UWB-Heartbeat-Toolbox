# 输入数据格式：NCJ29D6 CIR JSON

本项目原始 `RadarJsonReader` 读取具有以下关键字段的 JSON：

```json
{
  "RX_RADAR_CIR_CONFIGURATION": {"CIR_TAPS": 128},
  "RX_RADAR_CONTROL_CONFIGURATION": {"RX_ENBL": 2},
  "RADAR_NOISE_SUPPRESSION_CONFIGURATION": {
    "RNS_RX1": {"SELF_INTERFERENCE_TAP": 0},
    "RNS_RX2": {"SELF_INTERFERENCE_TAP": 20}
  },
  "CMD_RADAR_APPLICATION_CONFIG": {"FRAME_INTERVAL": 100000000},
  "cir_data (decimal)": [[0, 0, 0, 0]]
}
```
## 解码规则

- 第 `m` 帧共有 `CIR_TAPS` 个距离门。
- 每个距离门使用连续 4 字节：实部 2 字节（带符号、little-endian），虚部 2 字节（带符号、little-endian）。
- 得到形状为 `(帧数, CIR_TAPS)` 的复数 CIR 数组。
- `FRAME_INTERVAL` 单位为纳秒；采样率 `fs = 1/(FRAME_INTERVAL × 1e-9)`。
- 根据 `RX_ENBL` 选取 `RNS_RX1` 或 `RNS_RX2` 的 `SELF_INTERFERENCE_TAP`，距离轴 `range_axis_m=(tap-n0)*0.15`。

## 空场参考

建议额外提供在相同雷达参数和场景下录制的空场 JSON，通过 `--empty` 传入；本代码只检查空场文件的 `CIR_TAPS` 是否与测量文件一致，不会自动校验其它硬件参数，因此使用者需要自行保证两者条件匹配。

若没有空场文件，原程序使用当前测量数据前若干帧估计参考方差。这是一个退化方案，若参考片段本身有人或明显运动，会影响目标距离门评分。
