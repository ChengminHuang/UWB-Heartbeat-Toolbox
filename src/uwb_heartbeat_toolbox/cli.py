import argparse
from .pipeline import BreatheAndHeartSlidingWindowed

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sliding-window main-gate respiration and heartbeat extraction")
    parser.add_argument(
        "--filename",
        required=True,
        help="input radar JSON file",
    )
    parser.add_argument(
        "--empty",
        default="",
        help="empty/background JSON file; use an empty string to disable",
    )
    parser.add_argument("--range-min", type=float, default=0.3)
    parser.add_argument("--range-max", type=float, default=7.65)
    parser.add_argument("--window-sec", type=float, default=20.0)
    parser.add_argument("--hop-sec", type=float, default=1)
    parser.add_argument("--plot-every", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=2000.0)
    parser.add_argument("--vmd-k", type=int, nargs="+", default=[5])
    parser.add_argument("--vmd-alpha", type=float, nargs="+", default=[1000.0])
    parser.add_argument(
        "--vmd-plot-dir",
        default="",
        help="directory used to save VMD IMF figures; use an empty string to disable",
    )
    parser.add_argument(
        "--vmd-plot-format",
        default="tif",
        choices=["tif", "tiff", "svg", "pdf", "png", ""],
        help="output format for each VMD IMF figure",
    )
    parser.add_argument("--vmd-plot-show", action="store_true")
    parser.add_argument("--vmd-plot-normalize", action="store_true")
    parser.add_argument(
        "--complex-var-threshold",
        type=float,
        default=None,
        help="current-window complex-variance score threshold; default uses --threshold",
    )
    parser.add_argument(
        "--resp-null-amplitude",
        type=float,
        default=0.1,
        help="reference no-respiration sine amplitude used for window energy gating",
    )
    parser.add_argument(
        "--heart-null-amplitude",
        type=float,
        default=0.01,
        help="reference no-heartbeat sine amplitude used for window energy gating",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    empty_path = args.empty if args.empty else None
    processor = BreatheAndHeartSlidingWindowed(
        filename=args.filename,
        preferred_range_m=(args.range_min, args.range_max),
        empty_filename=empty_path,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        plot_every=args.plot_every,
        detection_threshold=args.threshold,
        vmd_k_options=tuple(args.vmd_k),
        vmd_alpha_options=tuple(args.vmd_alpha),
        vmd_plot_dir=args.vmd_plot_dir if args.vmd_plot_dir else None,
        vmd_plot_format=args.vmd_plot_format,
        vmd_plot_show=args.vmd_plot_show,
        vmd_plot_normalize=args.vmd_plot_normalize,
        complex_var_threshold=args.complex_var_threshold,
        resp_null_amplitude=args.resp_null_amplitude,
        heart_null_amplitude=args.heart_null_amplitude,
    )
    processor.run()



if __name__ == "__main__":
    main()
