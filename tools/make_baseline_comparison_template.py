import argparse
import csv
from pathlib import Path


EXPECTED_ROWS = [
    {"experiment": "Sketchy-U", "dataset": "sketchy", "baseline_method": "LPWM"},
    {"experiment": "Sketchy-A", "dataset": "sketchy", "baseline_method": "LPWM-Action"},
    {"experiment": "BAIR-U", "dataset": "bair", "baseline_method": "LPWM"},
    {"experiment": "Mario-U", "dataset": "mario", "baseline_method": "LPWM"},
    {"experiment": "Bridge-L", "dataset": "bridge", "baseline_method": "LPWM-Language"},
    {"experiment": "PandaPush", "dataset": "panda", "baseline_method": "LPWM-Image"},
    {"experiment": "OGBench-Scene", "dataset": "ogbench", "baseline_method": "LPWM-Image"},
]


FLOW_FIELDS = [
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


BASELINE_FIELDS = [
    "baseline_method",
    "baseline_source_table",
    "baseline_notes",
    "baseline_det_ssim",
    "baseline_det_psnr",
    "baseline_det_lpips",
    "baseline_sample_ssim",
    "baseline_sample_psnr",
    "baseline_sample_lpips",
    "baseline_sample_ctx_ssim",
    "baseline_sample_ctx_psnr",
    "baseline_sample_ctx_lpips",
    "baseline_sample_fvd",
]


DELTA_FIELDS = [
    "delta_det_ssim",
    "delta_det_psnr",
    "delta_det_lpips",
    "delta_sample_ssim",
    "delta_sample_psnr",
    "delta_sample_lpips",
    "delta_sample_ctx_ssim",
    "delta_sample_ctx_psnr",
    "delta_sample_ctx_lpips",
    "delta_sample_fvd",
]


def try_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_csv(path):
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def base_row_from_summary(row):
    result = {"experiment": row.get("experiment", "")}
    for field in FLOW_FIELDS:
        result[f"flow_{field}"] = row.get(field, "")
    result["baseline_method"] = ""
    result["baseline_source_table"] = ""
    result["baseline_notes"] = ""
    for field in BASELINE_FIELDS[3:]:
        result[field] = ""
    for field in DELTA_FIELDS:
        result[field] = ""
    return result


def expected_row_template(row):
    result = {"experiment": row["experiment"]}
    for field in FLOW_FIELDS:
        result[f"flow_{field}"] = ""
    result["flow_dataset"] = row["dataset"]
    result["baseline_method"] = row["baseline_method"]
    result["baseline_source_table"] = ""
    result["baseline_notes"] = ""
    for field in BASELINE_FIELDS[3:]:
        result[field] = ""
    for field in DELTA_FIELDS:
        result[field] = ""
    return result


def compute_deltas(row):
    pairs = [
        ("det_ssim", False),
        ("det_psnr", False),
        ("det_lpips", True),
        ("sample_ssim", False),
        ("sample_psnr", False),
        ("sample_lpips", True),
        ("sample_ctx_ssim", False),
        ("sample_ctx_psnr", False),
        ("sample_ctx_lpips", True),
        ("sample_fvd", True),
    ]
    for metric, lower_is_better in pairs:
        flow_value = try_float(row.get(f"flow_{metric}", ""))
        baseline_value = try_float(row.get(f"baseline_{metric}", ""))
        key = f"delta_{metric}"
        if flow_value is None or baseline_value is None:
            row[key] = ""
            continue
        row[key] = (baseline_value - flow_value) if lower_is_better else (flow_value - baseline_value)


def build_rows(flow_rows):
    if flow_rows:
        rows = [base_row_from_summary(row) for row in flow_rows]
    else:
        rows = [expected_row_template(row) for row in EXPECTED_ROWS]
    for row in rows:
        compute_deltas(row)
    return rows


def write_csv(rows, output_path):
    fieldnames = (
        ["experiment"]
        + [f"flow_{field}" for field in FLOW_FIELDS]
        + BASELINE_FIELDS
        + DELTA_FIELDS
    )
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(rows, output_path):
    headers = [
        "Experiment",
        "Baseline Method",
        "Flow Method",
        "Baseline Sample LPIPS",
        "Flow Sample LPIPS",
        "Delta Sample LPIPS",
        "Baseline Sample FVD",
        "Flow Sample FVD",
        "Delta Sample FVD",
        "Notes",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            values = [
                row["experiment"],
                row["baseline_method"],
                row["flow_method"],
                str(row["baseline_sample_lpips"]),
                str(row["flow_sample_lpips"]),
                str(row["delta_sample_lpips"]),
                str(row["baseline_sample_fvd"]),
                str(row["flow_sample_fvd"]),
                str(row["delta_sample_fvd"]),
                row["baseline_notes"],
            ]
            f.write("| " + " | ".join(values) + " |\n")


def main():
    parser = argparse.ArgumentParser(description="Make a baseline comparison template aligned to flow_exp_summary.csv.")
    parser.add_argument("--flow-summary-csv", default="./flow_exp_summary.csv",
                        help="Path to flow_exp_summary.csv. If missing, a blank experiment template is created.")
    parser.add_argument("--output-csv", default="./baseline_comparison_template.csv",
                        help="Output CSV path")
    parser.add_argument("--output-md", default="./baseline_comparison_template.md",
                        help="Output Markdown path")
    args = parser.parse_args()

    flow_rows = load_csv(Path(args.flow_summary_csv))
    rows = build_rows(flow_rows)
    write_csv(rows, args.output_csv)
    write_md(rows, args.output_md)
    print(f"wrote comparison csv: {Path(args.output_csv).resolve()}")
    print(f"wrote comparison md: {Path(args.output_md).resolve()}")


if __name__ == "__main__":
    main()
