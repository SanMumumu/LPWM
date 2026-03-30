import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(metrics, key):
    value = metrics.get(key, "")
    if isinstance(value, list):
        return value[-1] if value else ""
    return value


def build_row(experiment_name, run_dir):
    run_path = Path(run_dir).resolve()
    hparams = load_json(run_path / "hparams.json")
    det_metrics = load_json(run_path / "eval" / "metrics_test_det.json")
    sample_metrics = load_json(run_path / "eval" / "metrics_test_sample.json")
    sample_ctx_metrics = load_json(run_path / "eval" / "metrics_test_sample_ctx.json")
    sample_fvd = load_json(run_path / "eval" / "fvd_test_sample.json")

    checkpoint = ""
    save_dir = run_path / "saves"
    if save_dir.exists():
        best_lpips = sorted(save_dir.glob("*_best_lpips.pth"))
        best_any = sorted(save_dir.glob("*_best.pth"))
        all_ckpts = sorted(save_dir.glob("*.pth"))
        if best_lpips:
            checkpoint = str(best_lpips[-1].resolve())
        elif best_any:
            checkpoint = str(best_any[-1].resolve())
        elif all_ckpts:
            checkpoint = str(all_ckpts[-1].resolve())

    return {
        "experiment": experiment_name,
        "method": hparams.get("model_name", ""),
        "dataset": hparams.get("ds", ""),
        "root": hparams.get("root", ""),
        "num_epochs": hparams.get("num_epochs", ""),
        "run_dir": str(run_path),
        "checkpoint": checkpoint,
        "det_ssim": get_metric(det_metrics, "ssim"),
        "det_psnr": get_metric(det_metrics, "psnr"),
        "det_lpips": get_metric(det_metrics, "lpips"),
        "sample_ssim": get_metric(sample_metrics, "ssim"),
        "sample_psnr": get_metric(sample_metrics, "psnr"),
        "sample_lpips": get_metric(sample_metrics, "lpips"),
        "sample_ctx_ssim": get_metric(sample_ctx_metrics, "ssim"),
        "sample_ctx_psnr": get_metric(sample_ctx_metrics, "psnr"),
        "sample_ctx_lpips": get_metric(sample_ctx_metrics, "lpips"),
        "sample_fvd": get_metric(sample_fvd, "fvd"),
    }


def write_csv(rows, output_path):
    fieldnames = [
        "experiment",
        "method",
        "dataset",
        "root",
        "num_epochs",
        "run_dir",
        "checkpoint",
        "det_ssim",
        "det_psnr",
        "det_lpips",
        "sample_ssim",
        "sample_psnr",
        "sample_lpips",
        "sample_ctx_ssim",
        "sample_ctx_psnr",
        "sample_ctx_lpips",
        "sample_fvd",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows, output_path):
    headers = [
        "Experiment",
        "Method",
        "Dataset",
        "Epochs",
        "Det LPIPS",
        "Sample LPIPS",
        "Sample+Ctx LPIPS",
        "Sample FVD",
        "Run Dir",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = [
                row["experiment"],
                row["method"],
                row["dataset"],
                str(row["num_epochs"]),
                str(row["det_lpips"]),
                str(row["sample_lpips"]),
                str(row["sample_ctx_lpips"]),
                str(row["sample_fvd"]),
                row["run_dir"],
            ]
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Collect Flow-LPWM experiment results.")
    parser.add_argument("--run", action="append", default=[],
                        help="Experiment mapping in the form NAME=/abs/or/rel/run_dir")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument("--output-md", required=True, help="Output Markdown path")
    args = parser.parse_args()

    rows = []
    for item in args.run:
        if "=" not in item:
            raise ValueError(f"Invalid --run item: {item}")
        name, run_dir = item.split("=", 1)
        rows.append(build_row(name, run_dir))

    write_csv(rows, args.output_csv)
    write_md(rows, args.output_md)
    print(f"wrote summary csv: {Path(args.output_csv).resolve()}")
    print(f"wrote summary md: {Path(args.output_md).resolve()}")


if __name__ == "__main__":
    main()
