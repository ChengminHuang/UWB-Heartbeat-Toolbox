"""调用 UWB 心跳检测工具箱，并保存检测结果。"""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 不弹出绘图窗口，直接保存图片
import matplotlib.pyplot as plt

from uwb_heartbeat_toolbox import BreatheAndHeartSlidingWindowed


def save_results(processor, output_dir):
    """保存逐窗口检测结果和两张图片。"""

    # 创建输出目录
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = processor.window_results
    if not results:
        print("没有可保存的窗口结果")
        return

    # 1. 保存逐窗口检测数据 CSV
    csv_path = output_dir / "window_metrics.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(asdict(results[0]).keys())
        )
        writer.writeheader()

        for result in results:
            row = asdict(result)
            row["selected_imfs"] = json.dumps(row["selected_imfs"])
            writer.writerow(row)

    # 2. 绘制呼吸率和心率曲线
    times = [r.center_sec for r in results]
    resp_bpm = [r.resp_freq * 60 for r in results]
    heart_bpm = [r.heart_freq * 60 for r in results]

    plt.figure(figsize=(10, 4))
    plt.plot(times, resp_bpm, label="Respiration", linewidth=1.2)
    plt.plot(times, heart_bpm, label="Heartbeat", linewidth=1.2)

    plt.xlabel("Time (s)")
    plt.ylabel("Rate (BPM)")
    plt.title("Respiration and Heartbeat Rates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(figures_dir / "03_rate_curves.png", dpi=160)
    plt.close()

    # 3. 绘制 OLA 重建信号
    plt.figure(figsize=(10, 4))
    plt.plot(processor.axis_time, processor.resp_signal,
             label="Respiration", linewidth=1)
    plt.plot(processor.axis_time, processor.heart_signal,
             label="Heartbeat", linewidth=1)

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (rad)")
    plt.title("OLA Reconstructed Signals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(figures_dir / "04_ola_reconstruction.png", dpi=160)
    plt.close()

    print("Results saved to:", output_dir.resolve())


def main():
    # ==================== 用户配置 ====================
    input_file = "data/sample/measurement.json"
    empty_file = "data/sample/background.json"
    config_file = "configs/default.json"
    output_dir = "runs/case01"
    # ==================================================
    # 读取算法配置
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 创建算法对象
    processor = BreatheAndHeartSlidingWindowed(
        filename=input_file,
        empty_filename=empty_file,
        preferred_range_m=(
            config["range_min_m"],
            config["range_max_m"]
        ),
        window_sec=config["window_sec"],
        hop_sec=config["hop_sec"],
        plot_every=config["plot_every"],
        detection_threshold=config["detection_threshold"],
        complex_var_threshold=config["complex_var_threshold"],
        resp_null_amplitude=config["resp_null_amplitude"],
        heart_null_amplitude=config["heart_null_amplitude"],
        vmd_k_options=(config["vmd_k"],),
        vmd_alpha_options=(config["vmd_alpha"],),
        vmd_plot_dir=None,  # 不额外保存 VMD 图片
        vmd_plot_show=False
    )

    # 运行原始算法
    processor.run()

    # 保存结果
    save_results(processor, output_dir)


if __name__ == "__main__":
    main()
