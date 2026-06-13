#aggregate_result.py
import argparse
import glob
import itertools
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Basic utils
# ============================================================

Z_95 = 1.96


def safe_float(x, default=0.0):
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def safe_int(x, default=0):
    try:
        if x is None:
            return int(default)
        return int(x)
    except Exception:
        return int(default)


def is_finite_number(x):
    try:
        return pd.notna(x) and float(x) == float(x) and float(x) not in (float("inf"), float("-inf"))
    except Exception:
        return False


def clamp01(x):
    x = safe_float(x, 0.0)
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def safe_harmonic_mean(a, b):
    if not is_finite_number(a) or not is_finite_number(b):
        return float("nan")
    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * a * b / (a + b)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_head_train_mode(summary_path, summary):
    if "head_train_mode" in summary and summary["head_train_mode"] is not None:
        return str(summary["head_train_mode"])

    run_dir = os.path.dirname(summary_path)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
            cfg = manifest.get("config", {})
            if "head_train_mode" in cfg and cfg["head_train_mode"] is not None:
                return str(cfg["head_train_mode"])
        except Exception:
            pass

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            cfg = load_json(config_path)
            if "head_train_mode" in cfg and cfg["head_train_mode"] is not None:
                return str(cfg["head_train_mode"])
        except Exception:
            pass

    return "NA"


def extract_num_clients(summary_path, summary):
    if "num_clients" in summary and summary["num_clients"] is not None:
        try:
            return int(summary["num_clients"])
        except Exception:
            pass

    run_dir = os.path.dirname(summary_path)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
            cfg = manifest.get("config", {})
            if "num_clients" in cfg and cfg["num_clients"] is not None:
                return int(cfg["num_clients"])
        except Exception:
            pass

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            cfg = load_json(config_path)
            if "num_clients" in cfg and cfg["num_clients"] is not None:
                return int(cfg["num_clients"])
        except Exception:
            pass

    return -1


def extract_adapter_bottleneck(summary_path, summary):
    if "adapter_bottleneck" in summary and summary["adapter_bottleneck"] is not None:
        try:
            return int(summary["adapter_bottleneck"])
        except Exception:
            pass

    run_dir = os.path.dirname(summary_path)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
            cfg = manifest.get("config", {})
            if "adapter_bottleneck" in cfg and cfg["adapter_bottleneck"] is not None:
                return int(cfg["adapter_bottleneck"])
        except Exception:
            pass

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            cfg = load_json(config_path)
            if "adapter_bottleneck" in cfg and cfg["adapter_bottleneck"] is not None:
                return int(cfg["adapter_bottleneck"])
        except Exception:
            pass

    return 16


def extract_str_meta(summary_path, summary, key, default="NA"):
    if key in summary and summary[key] is not None:
        return str(summary[key])

    run_dir = os.path.dirname(summary_path)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
            cfg = manifest.get("config", {})
            if key in cfg and cfg[key] is not None:
                return str(cfg[key])
        except Exception:
            pass

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            cfg = load_json(config_path)
            if key in cfg and cfg[key] is not None:
                return str(cfg[key])
        except Exception:
            pass

    return default


def extract_bool_meta(summary_path, summary, key, default=False):
    if key in summary and summary[key] is not None:
        return bool(summary[key])

    run_dir = os.path.dirname(summary_path)

    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            manifest = load_json(manifest_path)
            cfg = manifest.get("config", {})
            if key in cfg and cfg[key] is not None:
                return bool(cfg[key])
        except Exception:
            pass

    config_path = os.path.join(run_dir, "config.json")
    if os.path.exists(config_path):
        try:
            cfg = load_json(config_path)
            if key in cfg and cfg[key] is not None:
                return bool(cfg[key])
        except Exception:
            pass

    return bool(default)


def load_last_metrics_line(metrics_path):
    last = None
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise RuntimeError("No metrics found in {}".format(metrics_path))
    return last


def resolve_metrics_path(summary_path, summary):
    metrics_path = summary.get("metrics_path", None)

    if not metrics_path:
        candidate = os.path.join(os.path.dirname(summary_path), "metrics.jsonl")
        return candidate if os.path.exists(candidate) else None

    if os.path.isabs(metrics_path):
        return metrics_path if os.path.exists(metrics_path) else None

    candidate_1 = os.path.join(os.getcwd(), metrics_path)
    if os.path.exists(candidate_1):
        return candidate_1

    candidate_2 = os.path.join(os.path.dirname(summary_path), metrics_path)
    if os.path.exists(candidate_2):
        return candidate_2

    candidate_3 = os.path.join(os.path.dirname(summary_path), os.path.basename(metrics_path))
    if os.path.exists(candidate_3):
        return candidate_3

    return None


def find_summary_files(root):
    return sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True))


def extract_run_name(summary_path):
    return os.path.basename(os.path.dirname(summary_path))


def normalize_group_columns(df, group_cols):
    df = df.copy()

    for col in group_cols:
        if col not in df.columns:
            continue

        if df[col].dtype == object:
            df[col] = df[col].fillna("NA")
        else:
            df[col] = df[col].fillna(-999999)

    return df


def denormalize_for_display(x):
    if not is_finite_number(x):
        return "NA"
    if float(x) == -999999:
        return "NA"
    return "{:.4f}".format(float(x))


def safe_std_value(x):
    if pd.isna(x):
        return 0.0
    return float(x)


def compute_ci95_half_width(std_value, n_value):
    std_value = safe_float(std_value, 0.0)
    n_value = safe_int(n_value, 0)
    if n_value <= 1:
        return 0.0
    return float(Z_95 * std_value / math.sqrt(float(n_value)))


def add_ci95_columns(df, metric_prefixes):
    df = df.copy()

    for prefix in metric_prefixes:
        mean_col = "{}_mean".format(prefix)
        std_col = "{}_std".format(prefix)

        if mean_col not in df.columns or std_col not in df.columns or "n_seeds" not in df.columns:
            continue

        ci_half_col = "{}_ci95_half".format(prefix)
        ci_low_col = "{}_ci95_low".format(prefix)
        ci_high_col = "{}_ci95_high".format(prefix)

        df[ci_half_col] = [
            compute_ci95_half_width(std_val, n_val)
            for std_val, n_val in zip(df[std_col], df["n_seeds"])
        ]
        df[ci_low_col] = df[mean_col] - df[ci_half_col]
        df[ci_high_col] = df[mean_col] + df[ci_half_col]

    return df


def format_mean_std(mean_value, std_value, digits=4):
    if pd.isna(mean_value):
        return "NaN"
    return ("{:." + str(digits) + "f} ± {:." + str(digits) + "f}").format(
        float(mean_value),
        safe_std_value(std_value),
    )


def format_mean_std_ci(mean_value, std_value, ci_low, ci_high, digits=4):
    if pd.isna(mean_value):
        return "NaN"
    return (
        ("{:." + str(digits) + "f} ± {:." + str(digits) + "f} [95% CI: {:." + str(digits) + "f}, {:." + str(digits) + "f}]")
        .format(
            float(mean_value),
            safe_std_value(std_value),
            float(ci_low),
            float(ci_high),
        )
    )


# ============================================================
# Significance testing helpers
# ============================================================

def variance_sample(values):
    values = [float(v) for v in values if is_finite_number(v)]
    n = len(values)
    if n <= 1:
        return 0.0
    mean_v = sum(values) / float(n)
    return sum((v - mean_v) ** 2 for v in values) / float(n - 1)


def std_sample(values):
    return math.sqrt(max(0.0, variance_sample(values)))


def exact_paired_sign_flip_pvalue(differences):
    """
    Two-sided exact sign-flip permutation p-value for paired differences.
    Appropriate for matched seeds and small n.
    """
    diffs = [float(d) for d in differences if is_finite_number(d)]
    n = len(diffs)
    if n == 0:
        return float("nan")
    if n == 1:
        return 1.0

    observed = abs(sum(diffs))
    total = 0
    extreme = 0

    for signs in itertools.product([-1.0, 1.0], repeat=n):
        stat = abs(sum(s * d for s, d in zip(signs, diffs)))
        total += 1
        if stat >= observed - 1e-12:
            extreme += 1

    return float(extreme / total) if total > 0 else float("nan")


def benjamini_hochberg_flags(p_values, alpha=0.05):
    indexed = [(idx, p) for idx, p in enumerate(p_values) if is_finite_number(p)]
    m = len(indexed)
    flags = [False] * len(p_values)

    if m == 0:
        return flags

    indexed_sorted = sorted(indexed, key=lambda x: x[1])
    threshold_rank = -1

    for rank, (orig_idx, p) in enumerate(indexed_sorted, start=1):
        crit = alpha * rank / float(m)
        if p <= crit:
            threshold_rank = rank

    if threshold_rank == -1:
        return flags

    for rank, (orig_idx, p) in enumerate(indexed_sorted, start=1):
        if rank <= threshold_rank:
            flags[orig_idx] = True

    return flags


# ============================================================
# Metric extraction
# ============================================================

def extract_final_accuracy(summary, last_metric):
    if "final_avg_accuracy" in summary:
        return safe_float(summary["final_avg_accuracy"])
    if "eval_seen_avg_accuracy" in last_metric:
        return safe_float(last_metric["eval_seen_avg_accuracy"])
    if "avg_acc_seen" in last_metric:
        return safe_float(last_metric["avg_acc_seen"])
    return 0.0


def extract_final_forgetting(summary, last_metric):
    if "final_avg_forgetting" in summary:
        return safe_float(summary["final_avg_forgetting"])

    if "task_level_avg_forgetting" in last_metric:
        return safe_float(last_metric["task_level_avg_forgetting"])

    if "avg_forgetting_seen" in last_metric:
        return safe_float(last_metric["avg_forgetting_seen"])

    if "final_avg_forgetting" in last_metric:
        return safe_float(last_metric["final_avg_forgetting"])

    return float("nan")


def extract_final_bwt(summary, last_metric):
    if "final_avg_bwt" in summary:
        return safe_float(summary["final_avg_bwt"])

    if "task_level_avg_bwt" in last_metric:
        return safe_float(last_metric["task_level_avg_bwt"])

    if "final_avg_bwt" in last_metric:
        return safe_float(last_metric["final_avg_bwt"])

    return float("nan")


def extract_final_fwt(summary, last_metric):
    if "final_avg_fwt" in summary:
        return safe_float(summary["final_avg_fwt"])

    if "final_avg_fwt" in last_metric:
        return safe_float(last_metric["final_avg_fwt"])

    return float("nan")


def extract_final_accuracy_drop_from_peak(summary, last_metric):
    if "final_avg_accuracy_drop_from_peak" in summary:
        return safe_float(summary["final_avg_accuracy_drop_from_peak"])

    if "task_level_avg_accuracy_drop_from_peak" in last_metric:
        return safe_float(last_metric["task_level_avg_accuracy_drop_from_peak"])

    if "final_avg_accuracy_drop_from_peak" in last_metric:
        return safe_float(last_metric["final_avg_accuracy_drop_from_peak"])

    return float("nan")


def extract_final_client_seen_accuracy_mean(summary, last_metric):
    if "final_client_seen_accuracy_mean" in summary:
        return safe_float(summary["final_client_seen_accuracy_mean"])
    if "client_seen_accuracy_mean" in last_metric:
        return safe_float(last_metric["client_seen_accuracy_mean"])
    return float("nan")


def extract_final_client_seen_accuracy_std(summary, last_metric):
    if "final_client_seen_accuracy_std" in summary:
        return safe_float(summary["final_client_seen_accuracy_std"])
    if "client_seen_accuracy_std" in last_metric:
        return safe_float(last_metric["client_seen_accuracy_std"])
    return float("nan")


def extract_final_client_seen_accuracy_min(summary, last_metric):
    if "final_client_seen_accuracy_min" in summary:
        return safe_float(summary["final_client_seen_accuracy_min"])
    if "client_seen_accuracy_min" in last_metric:
        return safe_float(last_metric["client_seen_accuracy_min"])
    return float("nan")


def extract_final_client_seen_accuracy_max(summary, last_metric):
    if "final_client_seen_accuracy_max" in summary:
        return safe_float(summary["final_client_seen_accuracy_max"])
    if "client_seen_accuracy_max" in last_metric:
        return safe_float(last_metric["client_seen_accuracy_max"])
    return float("nan")


def extract_replay_memory_mb(summary, last_metric):
    if "replay_memory_mb" in summary:
        return safe_float(summary["replay_memory_mb"])
    if "avg_replay_memory_mb" in summary:
        return safe_float(summary["avg_replay_memory_mb"])
    if "avg_replay_memory_mb" in last_metric:
        return safe_float(last_metric["avg_replay_memory_mb"])
    if "replay_memory_mb" in last_metric:
        return safe_float(last_metric["replay_memory_mb"])
    return 0.0


def extract_alpha(summary):
    if "alpha" not in summary:
        return float("nan")
    return safe_float(summary.get("alpha", float("nan")), default=float("nan"))


def extract_replay_flag(summary):
    replay = summary.get("replay", None)
    if replay is None:
        rpc = safe_int(summary.get("replay_per_class", 0), default=0)
        return bool(rpc > 0)
    return bool(replay)


# ============================================================
# Derived plasticity / stability metrics
# ============================================================

def compute_plasticity_score(final_accuracy):
    return clamp01(final_accuracy)


def compute_stability_score(final_forgetting):
    if not is_finite_number(final_forgetting):
        return float("nan")
    return clamp01(1.0 - float(final_forgetting))


def compute_stability_peak_score(final_accuracy_drop_from_peak):
    if not is_finite_number(final_accuracy_drop_from_peak):
        return float("nan")
    return clamp01(1.0 - float(final_accuracy_drop_from_peak))


# ============================================================
# Run collection
# ============================================================

def collect_runs(root):
    summary_files = find_summary_files(root)
    rows = []

    for summary_path in summary_files:
        try:
            summary = load_json(summary_path)
        except Exception as e:
            print("[WARN] Failed to read summary {}: {}".format(summary_path, e))
            continue

        metrics_path = resolve_metrics_path(summary_path, summary)
        if metrics_path is not None and os.path.exists(metrics_path):
            try:
                last_metric = load_last_metrics_line(metrics_path)
            except Exception as e:
                print("[WARN] Failed to read metrics.jsonl for {}: {}".format(summary_path, e))
                last_metric = {}
        else:
            print("[WARN] metrics.jsonl not found for run: {}".format(summary_path))
            last_metric = {}

        comm_bytes_total_run = safe_float(summary.get("comm_bytes_total_run", 0.0))
        wall_time_sec = safe_float(summary.get("wall_time_sec", 0.0))

        final_accuracy = extract_final_accuracy(summary, last_metric)
        final_forgetting = extract_final_forgetting(summary, last_metric)
        final_bwt = extract_final_bwt(summary, last_metric)
        final_fwt = extract_final_fwt(summary, last_metric)
        final_accuracy_drop_from_peak = extract_final_accuracy_drop_from_peak(summary, last_metric)

        plasticity_score = compute_plasticity_score(final_accuracy)
        stability_score = compute_stability_score(final_forgetting)
        stability_peak_score = compute_stability_peak_score(final_accuracy_drop_from_peak)
        plasticity_stability_hmean = safe_harmonic_mean(plasticity_score, stability_score)
        plasticity_stability_peak_hmean = safe_harmonic_mean(plasticity_score, stability_peak_score)

        row = {
            "run_name": extract_run_name(summary_path),
            "run_dir": os.path.dirname(summary_path),

            "dataset_name": summary.get("dataset_name", "UNKNOWN"),
            "method": summary.get("method", "UNKNOWN"),
            "fl_algo": summary.get("fl_algo", "UNKNOWN"),
            "head_train_mode": extract_head_train_mode(summary_path, summary),
            "adapter_bottleneck": extract_adapter_bottleneck(summary_path, summary),
            "trainable_scope": extract_str_meta(summary_path, summary, "trainable_scope", default="NA"),
            "evaluation_protocol": extract_str_meta(summary_path, summary, "evaluation_protocol", default="NA"),
            "backbone_pretrained": extract_bool_meta(summary_path, summary, "backbone_pretrained", default=False),
            "backbone_frozen": extract_bool_meta(summary_path, summary, "backbone_frozen", default=False),
            "task_id_known_at_inference": extract_bool_meta(summary_path, summary, "task_id_known_at_inference", default=True),

            "seed": safe_int(summary.get("seed", -1), default=-1),
            "num_clients": extract_num_clients(summary_path, summary),
            "tasks": safe_int(summary.get("tasks", 0), default=0),
            "rounds_per_task": safe_int(summary.get("rounds_per_task", 0), default=0),
            "local_epochs": safe_int(summary.get("local_epochs", 0), default=0),

            "alpha": extract_alpha(summary),
            "replay": extract_replay_flag(summary),
            "replay_per_class": safe_int(summary.get("replay_per_class", 0), default=0),
            "replay_lambda": safe_float(summary.get("replay_lambda", 0.0), default=0.0),
            "mu": safe_float(summary.get("mu", 0.0), default=0.0),

            "comm_bytes_total_run": comm_bytes_total_run,
            "upload_bytes_total_run": safe_float(summary.get("upload_bytes_total_run", 0.0)),
            "download_bytes_total_run": safe_float(summary.get("download_bytes_total_run", 0.0)),
            "wall_time_sec": wall_time_sec,

            "final_accuracy": final_accuracy,
            "final_forgetting": final_forgetting,
            "final_bwt": final_bwt,
            "final_fwt": final_fwt,
            "final_accuracy_drop_from_peak": final_accuracy_drop_from_peak,

            "plasticity_score": plasticity_score,
            "stability_score": stability_score,
            "stability_peak_score": stability_peak_score,
            "plasticity_stability_hmean": plasticity_stability_hmean,
            "plasticity_stability_peak_hmean": plasticity_stability_peak_hmean,

            "final_client_seen_accuracy_mean": extract_final_client_seen_accuracy_mean(summary, last_metric),
            "final_client_seen_accuracy_std": extract_final_client_seen_accuracy_std(summary, last_metric),
            "final_client_seen_accuracy_min": extract_final_client_seen_accuracy_min(summary, last_metric),
            "final_client_seen_accuracy_max": extract_final_client_seen_accuracy_max(summary, last_metric),

            "replay_memory_mb": extract_replay_memory_mb(summary, last_metric),

            "avg_power_w": safe_float(summary.get("avg_power_w", float("nan")), default=float("nan")),
            "peak_power_w": safe_float(summary.get("peak_power_w", float("nan")), default=float("nan")),
            "energy_j": safe_float(summary.get("energy_j", float("nan")), default=float("nan")),
            "avg_cpu_util": safe_float(summary.get("avg_cpu_util", float("nan")), default=float("nan")),
            "peak_cpu_util": safe_float(summary.get("peak_cpu_util", float("nan")), default=float("nan")),
            "avg_gpu_util": safe_float(summary.get("avg_gpu_util", float("nan")), default=float("nan")),
            "peak_gpu_util": safe_float(summary.get("peak_gpu_util", float("nan")), default=float("nan")),
            "avg_ram_mb": safe_float(summary.get("avg_ram_mb", float("nan")), default=float("nan")),
            "peak_ram_mb": safe_float(summary.get("peak_ram_mb", float("nan")), default=float("nan")),
            "avg_swap_mb": safe_float(summary.get("avg_swap_mb", float("nan")), default=float("nan")),
            "peak_swap_mb": safe_float(summary.get("peak_swap_mb", float("nan")), default=float("nan")),
            "tegrastats_found": bool(summary.get("tegrastats_found", False)),
            "tegrastats_sample_count": safe_int(summary.get("tegrastats_sample_count", 0), default=0),
            "power_source": str(summary.get("power_source", "NA")),

            "final_train_loss": safe_float(last_metric.get("avg_local_train_loss", 0.0)),
            "final_round_time_sec": safe_float(last_metric.get("round_time_sec", 0.0)),
        }

        row["comm_mb_total_run"] = comm_bytes_total_run / (1024.0 * 1024.0)
        row["wall_time_min"] = wall_time_sec / 60.0

        if row["comm_mb_total_run"] > 0:
            row["accuracy_per_comm_mb"] = row["final_accuracy"] / row["comm_mb_total_run"]
        else:
            row["accuracy_per_comm_mb"] = float("nan")

        rows.append(row)

    return rows


# ============================================================
# DataFrames
# ============================================================

def build_results_dataframe(root):
    rows = collect_runs(root)
    if not rows:
        raise RuntimeError("No experiment results found under: {}".format(root))

    df = pd.DataFrame(rows)

    required = [
        "dataset_name",
        "method",
        "fl_algo",
        "head_train_mode",
        "adapter_bottleneck",
        "trainable_scope",
        "evaluation_protocol",
        "backbone_pretrained",
        "backbone_frozen",
        "task_id_known_at_inference",
        "seed",
        "num_clients",
        "final_accuracy",
        "final_forgetting",
        "final_bwt",
        "final_fwt",
        "final_accuracy_drop_from_peak",
        "plasticity_score",
        "stability_score",
        "stability_peak_score",
        "plasticity_stability_hmean",
        "plasticity_stability_peak_hmean",
        "final_client_seen_accuracy_mean",
        "final_client_seen_accuracy_std",
        "final_client_seen_accuracy_min",
        "final_client_seen_accuracy_max",
        "accuracy_per_comm_mb",
        "comm_mb_total_run",
        "wall_time_min",
        "replay_memory_mb",
        "avg_power_w",
        "peak_power_w",
        "energy_j",
        "avg_cpu_util",
        "peak_cpu_util",
        "avg_gpu_util",
        "peak_gpu_util",
        "avg_ram_mb",
        "peak_ram_mb",
        "avg_swap_mb",
        "peak_swap_mb",
    ]
    for col in required:
        if col not in df.columns:
            if col in [
                "final_forgetting",
                "final_bwt",
                "final_fwt",
                "final_accuracy_drop_from_peak",
                "plasticity_score",
                "stability_score",
                "stability_peak_score",
                "plasticity_stability_hmean",
                "plasticity_stability_peak_hmean",
                "final_client_seen_accuracy_mean",
                "final_client_seen_accuracy_std",
                "final_client_seen_accuracy_min",
                "final_client_seen_accuracy_max",
                "accuracy_per_comm_mb",
                "avg_power_w",
                "peak_power_w",
                "energy_j",
                "avg_cpu_util",
                "peak_cpu_util",
                "avg_gpu_util",
                "peak_gpu_util",
                "avg_ram_mb",
                "peak_ram_mb",
                "avg_swap_mb",
                "peak_swap_mb",
            ]:
                df[col] = float("nan")
            elif col in ["head_train_mode", "trainable_scope", "evaluation_protocol"]:
                df[col] = "NA"
            elif col in ["num_clients", "adapter_bottleneck"]:
                df[col] = -1
            elif col in ["backbone_pretrained", "backbone_frozen", "task_id_known_at_inference"]:
                df[col] = False
            else:
                df[col] = 0.0

    return df


def filter_valid_forgetting(df):
    return df[df["final_forgetting"].apply(is_finite_number)].copy()


def build_summary_table(all_df):
    group_cols = [
        "dataset_name",
        "method",
        "fl_algo",
        "head_train_mode",
        "adapter_bottleneck",
        "trainable_scope",
        "evaluation_protocol",
        "backbone_pretrained",
        "backbone_frozen",
        "task_id_known_at_inference",
        "num_clients",
        "alpha",
        "replay",
        "replay_per_class",
        "mu",
    ]

    df_group = normalize_group_columns(all_df, group_cols)

    grouped = (
        df_group.groupby(group_cols)
        .agg(
            n_seeds=("seed", "count"),

            final_accuracy_mean=("final_accuracy", "mean"),
            final_accuracy_std=("final_accuracy", "std"),

            final_forgetting_mean=("final_forgetting", "mean"),
            final_forgetting_std=("final_forgetting", "std"),

            final_bwt_mean=("final_bwt", "mean"),
            final_bwt_std=("final_bwt", "std"),

            final_fwt_mean=("final_fwt", "mean"),
            final_fwt_std=("final_fwt", "std"),

            final_accuracy_drop_from_peak_mean=("final_accuracy_drop_from_peak", "mean"),
            final_accuracy_drop_from_peak_std=("final_accuracy_drop_from_peak", "std"),

            plasticity_score_mean=("plasticity_score", "mean"),
            plasticity_score_std=("plasticity_score", "std"),

            stability_score_mean=("stability_score", "mean"),
            stability_score_std=("stability_score", "std"),

            stability_peak_score_mean=("stability_peak_score", "mean"),
            stability_peak_score_std=("stability_peak_score", "std"),

            plasticity_stability_hmean_mean=("plasticity_stability_hmean", "mean"),
            plasticity_stability_hmean_std=("plasticity_stability_hmean", "std"),

            plasticity_stability_peak_hmean_mean=("plasticity_stability_peak_hmean", "mean"),
            plasticity_stability_peak_hmean_std=("plasticity_stability_peak_hmean", "std"),

            client_seen_accuracy_mean_mean=("final_client_seen_accuracy_mean", "mean"),
            client_seen_accuracy_mean_std=("final_client_seen_accuracy_mean", "std"),

            client_seen_accuracy_std_mean=("final_client_seen_accuracy_std", "mean"),
            client_seen_accuracy_std_std=("final_client_seen_accuracy_std", "std"),

            client_seen_accuracy_min_mean=("final_client_seen_accuracy_min", "mean"),
            client_seen_accuracy_min_std=("final_client_seen_accuracy_min", "std"),

            client_seen_accuracy_max_mean=("final_client_seen_accuracy_max", "mean"),
            client_seen_accuracy_max_std=("final_client_seen_accuracy_max", "std"),

            accuracy_per_comm_mb_mean=("accuracy_per_comm_mb", "mean"),
            accuracy_per_comm_mb_std=("accuracy_per_comm_mb", "std"),

            comm_mb_mean=("comm_mb_total_run", "mean"),
            comm_mb_std=("comm_mb_total_run", "std"),

            replay_memory_mb_mean=("replay_memory_mb", "mean"),
            replay_memory_mb_std=("replay_memory_mb", "std"),

            wall_time_min_mean=("wall_time_min", "mean"),
            wall_time_min_std=("wall_time_min", "std"),

            avg_power_w_mean=("avg_power_w", "mean"),
            avg_power_w_std=("avg_power_w", "std"),
            peak_power_w_mean=("peak_power_w", "mean"),
            peak_power_w_std=("peak_power_w", "std"),
            energy_j_mean=("energy_j", "mean"),
            energy_j_std=("energy_j", "std"),
            avg_cpu_util_mean=("avg_cpu_util", "mean"),
            avg_cpu_util_std=("avg_cpu_util", "std"),
            peak_cpu_util_mean=("peak_cpu_util", "mean"),
            peak_cpu_util_std=("peak_cpu_util", "std"),
            avg_gpu_util_mean=("avg_gpu_util", "mean"),
            avg_gpu_util_std=("avg_gpu_util", "std"),
            peak_gpu_util_mean=("peak_gpu_util", "mean"),
            peak_gpu_util_std=("peak_gpu_util", "std"),
            avg_ram_mb_mean=("avg_ram_mb", "mean"),
            avg_ram_mb_std=("avg_ram_mb", "std"),
            peak_ram_mb_mean=("peak_ram_mb", "mean"),
            peak_ram_mb_std=("peak_ram_mb", "std"),
        )
        .reset_index()
    )

    std_cols = [
        "final_accuracy_std",
        "final_forgetting_std",
        "final_bwt_std",
        "final_fwt_std",
        "final_accuracy_drop_from_peak_std",
        "plasticity_score_std",
        "stability_score_std",
        "stability_peak_score_std",
        "plasticity_stability_hmean_std",
        "plasticity_stability_peak_hmean_std",
        "client_seen_accuracy_mean_std",
        "client_seen_accuracy_std_std",
        "client_seen_accuracy_min_std",
        "client_seen_accuracy_max_std",
        "accuracy_per_comm_mb_std",
        "comm_mb_std",
        "replay_memory_mb_std",
        "wall_time_min_std",
        "avg_power_w_std",
        "peak_power_w_std",
        "energy_j_std",
        "avg_cpu_util_std",
        "peak_cpu_util_std",
        "avg_gpu_util_std",
        "peak_gpu_util_std",
        "avg_ram_mb_std",
        "peak_ram_mb_std",
    ]
    for col in std_cols:
        grouped[col] = grouped[col].fillna(0.0)

    grouped = add_ci95_columns(
        grouped,
        metric_prefixes=[
            "final_accuracy",
            "final_forgetting",
            "final_bwt",
            "final_fwt",
            "final_accuracy_drop_from_peak",
            "plasticity_score",
            "stability_score",
            "stability_peak_score",
            "plasticity_stability_hmean",
            "plasticity_stability_peak_hmean",
            "client_seen_accuracy_mean",
            "client_seen_accuracy_std",
            "client_seen_accuracy_min",
            "client_seen_accuracy_max",
            "accuracy_per_comm_mb",
            "comm_mb",
            "replay_memory_mb",
            "wall_time_min",
            "avg_power_w",
            "peak_power_w",
            "energy_j",
            "avg_cpu_util",
            "peak_cpu_util",
            "avg_gpu_util",
            "peak_gpu_util",
            "avg_ram_mb",
            "peak_ram_mb",
        ],
    )

    grouped = grouped.sort_values(
        [
            "dataset_name",
            "method",
            "fl_algo",
            "head_train_mode",
            "adapter_bottleneck",
            "backbone_pretrained",
            "num_clients",
            "alpha",
            "replay_per_class",
            "mu",
        ]
    ).reset_index(drop=True)

    return grouped


def build_pretty_summary(summary_df):
    df = summary_df.copy()

    def fmt_mean_std(mean_col, std_col, digits=4):
        vals = []
        for _, row in df.iterrows():
            vals.append(format_mean_std(row[mean_col], row[std_col], digits=digits))
        return vals

    def fmt_mean_std_ci(prefix, digits=4):
        vals = []
        mean_col = "{}_mean".format(prefix)
        std_col = "{}_std".format(prefix)
        ci_low_col = "{}_ci95_low".format(prefix)
        ci_high_col = "{}_ci95_high".format(prefix)

        for _, row in df.iterrows():
            vals.append(
                format_mean_std_ci(
                    row[mean_col],
                    row[std_col],
                    row[ci_low_col],
                    row[ci_high_col],
                    digits=digits,
                )
            )
        return vals

    pretty = pd.DataFrame({
        "dataset": df["dataset_name"],
        "method": df["method"],
        "fl_algo": df["fl_algo"],
        "head_train_mode": df["head_train_mode"],
        "adapter_bottleneck": df["adapter_bottleneck"],
        "trainable_scope": df["trainable_scope"],
        "evaluation_protocol": df["evaluation_protocol"],
        "backbone_pretrained": df["backbone_pretrained"],
        "backbone_frozen": df["backbone_frozen"],
        "task_id_known_at_inference": df["task_id_known_at_inference"],
        "num_clients": df["num_clients"],
        "alpha": [denormalize_for_display(x) for x in df["alpha"]],
        "replay": df["replay"],
        "replay_per_class": df["replay_per_class"],
        "mu": [denormalize_for_display(x) for x in df["mu"]],
        "n_seeds": df["n_seeds"],

        "final_accuracy": fmt_mean_std("final_accuracy_mean", "final_accuracy_std", digits=4),
        "final_accuracy_ci95": fmt_mean_std_ci("final_accuracy", digits=4),

        "final_forgetting": fmt_mean_std("final_forgetting_mean", "final_forgetting_std", digits=4),
        "final_forgetting_ci95": fmt_mean_std_ci("final_forgetting", digits=4),

        "final_bwt": fmt_mean_std("final_bwt_mean", "final_bwt_std", digits=4),
        "final_bwt_ci95": fmt_mean_std_ci("final_bwt", digits=4),

        "final_fwt": fmt_mean_std("final_fwt_mean", "final_fwt_std", digits=4),
        "final_fwt_ci95": fmt_mean_std_ci("final_fwt", digits=4),

        "peak_drop": fmt_mean_std(
            "final_accuracy_drop_from_peak_mean",
            "final_accuracy_drop_from_peak_std",
            digits=4,
        ),
        "peak_drop_ci95": fmt_mean_std_ci("final_accuracy_drop_from_peak", digits=4),

        "plasticity": fmt_mean_std("plasticity_score_mean", "plasticity_score_std", digits=4),
        "plasticity_ci95": fmt_mean_std_ci("plasticity_score", digits=4),

        "stability": fmt_mean_std("stability_score_mean", "stability_score_std", digits=4),
        "stability_ci95": fmt_mean_std_ci("stability_score", digits=4),

        "stability_peak": fmt_mean_std("stability_peak_score_mean", "stability_peak_score_std", digits=4),
        "stability_peak_ci95": fmt_mean_std_ci("stability_peak_score", digits=4),

        "plasticity_stability_hmean": fmt_mean_std(
            "plasticity_stability_hmean_mean",
            "plasticity_stability_hmean_std",
            digits=4,
        ),
        "plasticity_stability_hmean_ci95": fmt_mean_std_ci("plasticity_stability_hmean", digits=4),

        "plasticity_stability_peak_hmean": fmt_mean_std(
            "plasticity_stability_peak_hmean_mean",
            "plasticity_stability_peak_hmean_std",
            digits=4,
        ),
        "plasticity_stability_peak_hmean_ci95": fmt_mean_std_ci("plasticity_stability_peak_hmean", digits=4),

        "client_mean_acc": fmt_mean_std(
            "client_seen_accuracy_mean_mean",
            "client_seen_accuracy_mean_std",
            digits=4,
        ),
        "client_mean_acc_ci95": fmt_mean_std_ci("client_seen_accuracy_mean", digits=4),

        "client_acc_std": fmt_mean_std(
            "client_seen_accuracy_std_mean",
            "client_seen_accuracy_std_std",
            digits=4,
        ),
        "client_acc_std_ci95": fmt_mean_std_ci("client_seen_accuracy_std", digits=4),

        "client_worst_acc": fmt_mean_std(
            "client_seen_accuracy_min_mean",
            "client_seen_accuracy_min_std",
            digits=4,
        ),
        "client_worst_acc_ci95": fmt_mean_std_ci("client_seen_accuracy_min", digits=4),

        "client_best_acc": fmt_mean_std(
            "client_seen_accuracy_max_mean",
            "client_seen_accuracy_max_std",
            digits=4,
        ),
        "client_best_acc_ci95": fmt_mean_std_ci("client_seen_accuracy_max", digits=4),

        "acc_per_comm_mb": fmt_mean_std(
            "accuracy_per_comm_mb_mean",
            "accuracy_per_comm_mb_std",
            digits=4,
        ),
        "acc_per_comm_mb_ci95": fmt_mean_std_ci("accuracy_per_comm_mb", digits=4),

        "communication_mb": ["{:.2f}".format(x) for x in df["comm_mb_mean"]],
        "communication_mb_ci95": [
            "[95% CI: {:.2f}, {:.2f}]".format(low, high)
            for low, high in zip(df["comm_mb_ci95_low"], df["comm_mb_ci95_high"])
        ],

        "runtime_min": ["{:.2f}".format(x) for x in df["wall_time_min_mean"]],
        "runtime_min_ci95": [
            "[95% CI: {:.2f}, {:.2f}]".format(low, high)
            for low, high in zip(df["wall_time_min_ci95_low"], df["wall_time_min_ci95_high"])
        ],

        "avg_power_w": fmt_mean_std("avg_power_w_mean", "avg_power_w_std", digits=3),
        "avg_power_w_ci95": fmt_mean_std_ci("avg_power_w", digits=3),

        "peak_power_w": fmt_mean_std("peak_power_w_mean", "peak_power_w_std", digits=3),
        "peak_power_w_ci95": fmt_mean_std_ci("peak_power_w", digits=3),

        "energy_j": fmt_mean_std("energy_j_mean", "energy_j_std", digits=3),
        "energy_j_ci95": fmt_mean_std_ci("energy_j", digits=3),

        "avg_cpu_util": fmt_mean_std("avg_cpu_util_mean", "avg_cpu_util_std", digits=3),
        "avg_cpu_util_ci95": fmt_mean_std_ci("avg_cpu_util", digits=3),

        "avg_gpu_util": fmt_mean_std("avg_gpu_util_mean", "avg_gpu_util_std", digits=3),
        "avg_gpu_util_ci95": fmt_mean_std_ci("avg_gpu_util", digits=3),

        "peak_ram_mb": fmt_mean_std("peak_ram_mb_mean", "peak_ram_mb_std", digits=2),
        "peak_ram_mb_ci95": fmt_mean_std_ci("peak_ram_mb", digits=2),
    })

    return pretty


# ============================================================
# Pairwise significance testing
# ============================================================

def build_pairwise_significance_table(all_df):
    """
    Compare groups pairwise within the same:
      dataset_name, head_train_mode, adapter_bottleneck, backbone_pretrained,
      num_clients, alpha, replay, replay_per_class, mu
    using matched seeds.
    """
    metric_specs = [
        ("final_accuracy", "higher"),
        ("final_forgetting", "lower"),
        ("final_bwt", "higher"),
        ("final_fwt", "higher"),
        ("final_accuracy_drop_from_peak", "lower"),
        ("plasticity_score", "higher"),
        ("stability_score", "higher"),
        ("plasticity_stability_hmean", "higher"),
        ("avg_power_w", "lower"),
        ("energy_j", "lower"),
        ("avg_cpu_util", "lower"),
        ("avg_gpu_util", "lower"),
        ("peak_ram_mb", "lower"),
    ]

    compare_context_cols = [
        "dataset_name",
        "head_train_mode",
        "adapter_bottleneck",
        "backbone_pretrained",
        "num_clients",
        "alpha",
        "replay",
        "replay_per_class",
        "mu",
    ]

    rows = []
    df = normalize_group_columns(all_df, compare_context_cols)

    for context_values, ctx_df in df.groupby(compare_context_cols):
        ctx_df = ctx_df.copy()

        candidate_groups = []
        for (method, fl_algo), sub in ctx_df.groupby(["method", "fl_algo"]):
            candidate_groups.append({
                "method": method,
                "fl_algo": fl_algo,
                "sub": sub.copy(),
            })

        if len(candidate_groups) < 2:
            continue

        for g1, g2 in itertools.combinations(candidate_groups, 2):
            s1 = g1["sub"].set_index("seed")
            s2 = g2["sub"].set_index("seed")

            common_seeds = sorted(set(s1.index.tolist()) & set(s2.index.tolist()))
            if len(common_seeds) == 0:
                continue

            base_row = {
                "dataset_name": context_values[0],
                "head_train_mode": context_values[1],
                "adapter_bottleneck": context_values[2],
                "backbone_pretrained": context_values[3],
                "num_clients": context_values[4],
                "alpha": context_values[5],
                "replay": context_values[6],
                "replay_per_class": context_values[7],
                "mu": context_values[8],
                "method_a": g1["method"],
                "fl_algo_a": g1["fl_algo"],
                "method_b": g2["method"],
                "fl_algo_b": g2["fl_algo"],
                "label_a": "{}|{}".format(g1["method"], g1["fl_algo"]),
                "label_b": "{}|{}".format(g2["method"], g2["fl_algo"]),
                "n_common_seeds": len(common_seeds),
                "common_seeds": ",".join(str(x) for x in common_seeds),
            }

            for metric_name, better_direction in metric_specs:
                vals_a = []
                vals_b = []
                diffs = []

                for seed in common_seeds:
                    va = s1.loc[seed, metric_name]
                    vb = s2.loc[seed, metric_name]

                    if isinstance(va, pd.Series):
                        va = va.iloc[0]
                    if isinstance(vb, pd.Series):
                        vb = vb.iloc[0]

                    if not is_finite_number(va) or not is_finite_number(vb):
                        continue

                    va = float(va)
                    vb = float(vb)
                    vals_a.append(va)
                    vals_b.append(vb)
                    diffs.append(va - vb)

                metric_row = dict(base_row)
                metric_row["metric"] = metric_name
                metric_row["better_direction"] = better_direction
                metric_row["n_pairs_used"] = len(diffs)

                if len(diffs) == 0:
                    metric_row["mean_a"] = float("nan")
                    metric_row["mean_b"] = float("nan")
                    metric_row["mean_diff_a_minus_b"] = float("nan")
                    metric_row["std_diff"] = float("nan")
                    metric_row["effect_size_paired_dz"] = float("nan")
                    metric_row["p_value"] = float("nan")
                    metric_row["significant_p_lt_0_05"] = False
                    metric_row["winner"] = "NA"
                    rows.append(metric_row)
                    continue

                mean_a = sum(vals_a) / float(len(vals_a))
                mean_b = sum(vals_b) / float(len(vals_b))
                mean_diff = sum(diffs) / float(len(diffs))
                std_diff = std_sample(diffs)

                if std_diff > 0:
                    effect_size = mean_diff / std_diff
                else:
                    effect_size = 0.0

                p_value = exact_paired_sign_flip_pvalue(diffs)

                winner = "tie"
                if better_direction == "higher":
                    if mean_diff > 0:
                        winner = "A"
                    elif mean_diff < 0:
                        winner = "B"
                else:
                    if mean_diff < 0:
                        winner = "A"
                    elif mean_diff > 0:
                        winner = "B"

                metric_row["mean_a"] = mean_a
                metric_row["mean_b"] = mean_b
                metric_row["mean_diff_a_minus_b"] = mean_diff
                metric_row["std_diff"] = std_diff
                metric_row["effect_size_paired_dz"] = effect_size
                metric_row["p_value"] = p_value
                metric_row["significant_p_lt_0_05"] = bool(is_finite_number(p_value) and p_value < 0.05)
                metric_row["winner"] = winner
                rows.append(metric_row)

    sig_df = pd.DataFrame(rows)
    if sig_df.empty:
        return sig_df

    sig_df["fdr_significant_bh_0_05"] = False
    for metric_name, sub_idx in sig_df.groupby("metric").groups.items():
        idx_list = list(sub_idx)
        pvals = sig_df.loc[idx_list, "p_value"].tolist()
        flags = benjamini_hochberg_flags(pvals, alpha=0.05)
        sig_df.loc[idx_list, "fdr_significant_bh_0_05"] = flags

    sig_df = sig_df.sort_values(
        [
            "dataset_name",
            "head_train_mode",
            "adapter_bottleneck",
            "backbone_pretrained",
            "num_clients",
            "alpha",
            "replay_per_class",
            "mu",
            "metric",
            "method_a",
            "fl_algo_a",
            "method_b",
            "fl_algo_b",
        ]
    ).reset_index(drop=True)

    return sig_df


# ============================================================
# Saving
# ============================================================

def save_outputs(all_df, summary_df, out_dir, pairwise_sig_df=None):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pretty_df = build_pretty_summary(summary_df)

    all_runs_csv = os.path.join(out_dir, "all_runs.csv")
    summary_csv = os.path.join(out_dir, "summary_grouped.csv")
    summary_pretty_csv = os.path.join(out_dir, "summary_pretty.csv")
    missing_forgetting_csv = os.path.join(out_dir, "runs_missing_forgetting.csv")
    pairwise_sig_csv = os.path.join(out_dir, "pairwise_significance.csv")

    all_df.to_csv(all_runs_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    pretty_df.to_csv(summary_pretty_csv, index=False)

    missing_forgetting_df = all_df[~all_df["final_forgetting"].apply(is_finite_number)].copy()
    missing_forgetting_df.to_csv(missing_forgetting_csv, index=False)

    if pairwise_sig_df is None:
        pairwise_sig_df = pd.DataFrame()
    pairwise_sig_df.to_csv(pairwise_sig_csv, index=False)

    return all_runs_csv, summary_csv, summary_pretty_csv, missing_forgetting_csv, pairwise_sig_csv


# ============================================================
# Plots
# ============================================================

METHOD_ORDER = ["FULL_FEDAVG", "SHARED_ADAPTER", "TASK_ADAPTER", "TRUE_FEDAVG"]
METHOD_LABELS = {
    "FULL_FEDAVG": "Head-only",
    "SHARED_ADAPTER": "Shared Adapter",
    "TASK_ADAPTER": "Task Adapter",
    "TRUE_FEDAVG": "Full-model",
}
METHOD_MARKERS = {
    "FULL_FEDAVG": "s",
    "SHARED_ADAPTER": "o",
    "TASK_ADAPTER": "^",
    "TRUE_FEDAVG": "D",
}
DATASET_LABELS = {
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "digit5": "Digit5",
}


def dataset_display_name(dataset):
    return DATASET_LABELS.get(str(dataset).lower(), str(dataset))


def method_sort_key(method_name):
    if method_name in METHOD_ORDER:
        return METHOD_ORDER.index(method_name)
    return len(METHOD_ORDER) + 1


def get_method_label(method_name):
    return METHOD_LABELS.get(str(method_name), str(method_name))


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def journal_savefig(fig, out_path_no_ext):
    fig.savefig(out_path_no_ext + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(out_path_no_ext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def sort_for_display(sub):
    sub = sub.copy()
    sub["_method_order"] = sub["method"].apply(method_sort_key)
    sub = sub.sort_values(["_method_order", "method", "fl_algo"]).reset_index(drop=True)
    return sub


def build_group_label(row):
    method = get_method_label(row["method"])
    fl_algo = str(row["fl_algo"])
    if fl_algo == "FEDAVG":
        return method
    return "{} ({})".format(method, fl_algo)


def style_axes(ax):
    ax.grid(True, alpha=0.3, linewidth=0.7)
    ax.set_axisbelow(True)


def errorbar_kwargs_for_method(method_name):
    return {
        "fmt": METHOD_MARKERS.get(method_name, "o"),
        "markersize": 7,
        "capsize": 3,
        "linewidth": 1.6,
        "linestyle": "None",
        "label": get_method_label(method_name),
    }


def save_plot_accuracy_vs_forgetting(summary_df, out_dir):
    ensure_dir(out_dir)

    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    for dataset in datasets:
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        if sub.empty:
            continue

        sub = sub[
            sub["final_accuracy_mean"].apply(is_finite_number) &
            sub["final_forgetting_mean"].apply(is_finite_number)
        ].copy()
        if sub.empty:
            print("[WARN] Skipping journal trade-off plot for {} because no finite values exist.".format(dataset))
            continue

        sub = sort_for_display(sub)
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

        for _, row in sub.iterrows():
            axes[0].errorbar(
                row["final_forgetting_mean"],
                row["final_accuracy_mean"],
                xerr=safe_std_value(row.get("final_forgetting_std", 0.0)),
                yerr=safe_std_value(row.get("final_accuracy_std", 0.0)),
                **errorbar_kwargs_for_method(row["method"])
            )
        axes[0].set_xlabel("Final Forgetting")
        axes[0].set_ylabel("Final Accuracy")
        axes[0].set_title("(a) Accuracy vs Forgetting")
        style_axes(axes[0])

        use_runtime = str(dataset).lower() == "digit5"
        x_col = "wall_time_min_mean" if use_runtime else "comm_mb_mean"
        x_std_col = "wall_time_min_std" if use_runtime else "comm_mb_std"
        x_label = "Runtime (min)" if use_runtime else "Communication (MB, log scale)"
        title_label = "(b) Accuracy vs Runtime" if use_runtime else "(b) Accuracy vs Communication"

        sub2 = sub[sub[x_col].apply(is_finite_number)].copy()
        for _, row in sub2.iterrows():
            axes[1].errorbar(
                row[x_col],
                row["final_accuracy_mean"],
                yerr=safe_std_value(row.get("final_accuracy_std", 0.0)),
                xerr=safe_std_value(row.get(x_std_col, 0.0)),
                **errorbar_kwargs_for_method(row["method"])
            )

        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("Final Accuracy")
        axes[1].set_title(title_label)
        if not use_runtime:
            positive_x = [float(v) for v in sub2[x_col].tolist() if is_finite_number(v) and float(v) > 0]
            if positive_x:
                axes[1].set_xscale("log")
        style_axes(axes[1])

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    uniq_handles.append(h)
                    uniq_labels.append(l)
                    seen.add(l)
            fig.legend(uniq_handles, uniq_labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))

        fig.suptitle(dataset_display_name(dataset), y=1.02, fontsize=12)
        fig.tight_layout(rect=(0, 0.06, 1, 0.98))
        journal_savefig(fig, os.path.join(out_dir, "{}_accuracy_vs_forgetting".format(dataset)))


def save_plot_accuracy_vs_energy(summary_df, out_dir):
    ensure_dir(out_dir)

    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    for dataset in datasets:
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        if sub.empty:
            continue

        sub = sub[
            sub["final_accuracy_mean"].apply(is_finite_number) &
            sub["energy_j_mean"].apply(is_finite_number)
        ].copy()
        if sub.empty:
            continue

        sub = sort_for_display(sub)
        fig, ax = plt.subplots(figsize=(5.6, 4.3))
        for _, row in sub.iterrows():
            ax.errorbar(
                row["energy_j_mean"],
                row["final_accuracy_mean"],
                xerr=safe_std_value(row.get("energy_j_std", 0.0)),
                yerr=safe_std_value(row.get("final_accuracy_std", 0.0)),
                **errorbar_kwargs_for_method(row["method"])
            )

        ax.set_xlabel("Energy (J)")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("{}: Accuracy vs Energy".format(dataset_display_name(dataset)))
        style_axes(ax)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    uniq_handles.append(h)
                    uniq_labels.append(l)
                    seen.add(l)
            ax.legend(uniq_handles, uniq_labels, frameon=False, fontsize=9)

        fig.tight_layout()
        journal_savefig(fig, os.path.join(out_dir, "{}_accuracy_vs_energy".format(dataset)))


def save_plot_accuracy_vs_fairness(summary_df, out_dir):
    ensure_dir(out_dir)

    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    for dataset in datasets:
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        if sub.empty:
            continue

        sub = sub[
            sub["final_accuracy_mean"].apply(is_finite_number) &
            sub["client_seen_accuracy_std_mean"].apply(is_finite_number)
        ].copy()
        if sub.empty:
            print("[WARN] Skipping accuracy-vs-fairness plot for {} because no finite fairness values exist.".format(dataset))
            continue

        sub = sort_for_display(sub)
        fig, ax = plt.subplots(figsize=(5.4, 4.3))
        for _, row in sub.iterrows():
            ax.errorbar(
                row["client_seen_accuracy_std_mean"],
                row["final_accuracy_mean"],
                xerr=safe_std_value(row.get("client_seen_accuracy_std_std", 0.0)),
                yerr=safe_std_value(row.get("final_accuracy_std", 0.0)),
                **errorbar_kwargs_for_method(row["method"])
            )

        ax.set_xlabel("Client Accuracy Std (fairness disparity)")
        ax.set_ylabel("Final Accuracy")
        ax.set_title("{}: Accuracy vs Fairness".format(dataset_display_name(dataset)))
        style_axes(ax)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            seen = set()
            uniq_handles, uniq_labels = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    uniq_handles.append(h)
                    uniq_labels.append(l)
                    seen.add(l)
            ax.legend(uniq_handles, uniq_labels, frameon=False, fontsize=9)

        fig.tight_layout()
        journal_savefig(fig, os.path.join(out_dir, "{}_accuracy_vs_fairness".format(dataset)))


def save_plot_bar_forgetting(summary_df, out_dir):
    ensure_dir(out_dir)

    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    for dataset in datasets:
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        if sub.empty:
            continue

        sub = sub[sub["final_forgetting_mean"].apply(is_finite_number)].copy()
        if sub.empty:
            print("[WARN] Skipping forgetting bar plot for {} because no finite forgetting values exist.".format(dataset))
            continue

        sub = sort_for_display(sub)
        labels = [build_group_label(row) for _, row in sub.iterrows()]
        x = list(range(len(labels)))

        fig, ax = plt.subplots(figsize=(8.4, 4.3))
        ax.bar(
            x,
            sub["final_forgetting_mean"],
            yerr=sub["final_forgetting_std"],
            capsize=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Final Forgetting")
        ax.set_title("{}: Forgetting Comparison".format(dataset_display_name(dataset)))
        style_axes(ax)
        fig.tight_layout()
        journal_savefig(fig, os.path.join(out_dir, "{}_forgetting_bar".format(dataset)))


def save_journal_small_multiples(summary_df, out_dir):
    ensure_dir(out_dir)
    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    if len(datasets) < 2:
        return

    datasets = [d for d in datasets if not summary_df[summary_df["dataset_name"] == d].empty]
    if len(datasets) < 2:
        return

    fig, axes = plt.subplots(2, len(datasets), figsize=(4.0 * len(datasets), 6.0), squeeze=False)

    for col, dataset in enumerate(datasets):
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        sub = sub[
            sub["final_accuracy_mean"].apply(is_finite_number) &
            sub["final_forgetting_mean"].apply(is_finite_number)
        ].copy()
        if sub.empty:
            continue
        sub = sort_for_display(sub)

        labels = [get_method_label(m) for m in sub["method"].tolist()]
        xs = list(range(len(labels)))

        axes[0, col].plot(xs, sub["final_accuracy_mean"], marker="o", linewidth=1.8)
        axes[0, col].set_title(dataset_display_name(dataset))
        axes[0, col].set_xticks(xs)
        axes[0, col].set_xticklabels(labels, rotation=20, ha="right")
        if col == 0:
            axes[0, col].set_ylabel("Final Accuracy")
        style_axes(axes[0, col])

        axes[1, col].plot(xs, sub["final_forgetting_mean"], marker="o", linewidth=1.8)
        axes[1, col].set_xticks(xs)
        axes[1, col].set_xticklabels(labels, rotation=20, ha="right")
        if col == 0:
            axes[1, col].set_ylabel("Final Forgetting")
        style_axes(axes[1, col])

    fig.suptitle("Main benchmark comparison across datasets", y=1.01, fontsize=12)
    fig.tight_layout()
    journal_savefig(fig, os.path.join(out_dir, "journal_small_multiples_overview"))


def save_journal_heterogeneity_sweep(summary_df, out_dir):
    ensure_dir(out_dir)
    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    for dataset in datasets:
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        if sub.empty:
            continue

        alpha_values = [x for x in sorted(sub["alpha"].dropna().unique()) if is_finite_number(x)]
        if len(alpha_values) < 2:
            continue

        methods_present = sorted(sub["method"].dropna().unique(), key=method_sort_key)
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)

        for method_name in methods_present:
            ms = sub[sub["method"] == method_name].copy()
            ms = ms[ms["alpha"].apply(is_finite_number)].sort_values("alpha", ascending=False)
            if len(ms) < 2:
                continue

            xs = [float(v) for v in ms["alpha"].tolist()]
            ys_acc = [float(v) for v in ms["final_accuracy_mean"].tolist()]
            ys_forget = [float(v) for v in ms["final_forgetting_mean"].tolist()]

            axes[0].plot(xs, ys_acc, marker=METHOD_MARKERS.get(method_name, "o"), linewidth=1.8, label=get_method_label(method_name))
            axes[1].plot(xs, ys_forget, marker=METHOD_MARKERS.get(method_name, "o"), linewidth=1.8, label=get_method_label(method_name))

        axes[0].set_title("(a) Accuracy under heterogeneity")
        axes[1].set_title("(b) Forgetting under heterogeneity")
        axes[0].set_xlabel("Dirichlet alpha")
        axes[1].set_xlabel("Dirichlet alpha")
        axes[0].set_ylabel("Final Accuracy")
        axes[1].set_ylabel("Final Forgetting")
        for ax in axes:
            style_axes(ax)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.suptitle("{} heterogeneity sensitivity".format(dataset_display_name(dataset)), y=1.02, fontsize=12)
        fig.tight_layout(rect=(0, 0.06, 1, 0.98))
        journal_savefig(fig, os.path.join(out_dir, "{}_heterogeneity_sweep".format(dataset)))


def save_journal_bottleneck_sweep(summary_df, out_dir):
    ensure_dir(out_dir)
    datasets = sorted(summary_df["dataset_name"].dropna().unique())
    for dataset in datasets:
        sub = summary_df[summary_df["dataset_name"] == dataset].copy()
        if sub.empty:
            continue

        sub = sub[
            sub["adapter_bottleneck"].apply(is_finite_number) &
            sub["final_accuracy_mean"].apply(is_finite_number) &
            sub["comm_mb_mean"].apply(is_finite_number)
        ].copy()

        if len(sorted(sub["adapter_bottleneck"].dropna().unique())) < 2:
            continue

        if "TASK_ADAPTER" in sub["method"].tolist():
            sub = sub[sub["method"] == "TASK_ADAPTER"].copy()

        sub = sub.sort_values("adapter_bottleneck")
        xs = [int(v) for v in sub["adapter_bottleneck"].tolist()]
        if len(sorted(set(xs))) < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
        axes[0].errorbar(
            xs,
            sub["final_accuracy_mean"],
            yerr=sub["final_accuracy_std"],
            marker="o",
            linewidth=1.8,
            capsize=3,
        )
        axes[0].set_title("(a) Bottleneck vs Accuracy")
        axes[0].set_xlabel("Adapter bottleneck")
        axes[0].set_ylabel("Final Accuracy")
        style_axes(axes[0])

        axes[1].plot(xs, sub["comm_mb_mean"], marker="s", linewidth=1.8, label="Communication (MB)")
        axes[1].plot(xs, sub["wall_time_min_mean"], marker="^", linewidth=1.6, label="Runtime (min)")
        axes[1].set_title("(b) Bottleneck vs Cost")
        axes[1].set_xlabel("Adapter bottleneck")
        axes[1].set_ylabel("Value")
        style_axes(axes[1])
        axes[1].legend(frameon=False)

        fig.suptitle("{} bottleneck ablation".format(dataset_display_name(dataset)), y=1.02, fontsize=12)
        fig.tight_layout()
        journal_savefig(fig, os.path.join(out_dir, "{}_bottleneck_sweep".format(dataset)))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Aggregate FCL experiment results.")
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root experiment directory containing nested summary.json files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="analysis",
        help="Directory where CSVs and plots will be saved.",
    )
    parser.add_argument(
        "--only_valid_forgetting",
        action="store_true",
        help="Use only runs with non-NaN final_forgetting for summary tables and plots.",
    )
    args = parser.parse_args()

    print("[INFO] Scanning experiments under: {}".format(args.root))
    all_df = build_results_dataframe(args.root)

    missing_forgetting = int(all_df["final_forgetting"].isna().sum())
    print("[INFO] Total runs found: {}".format(len(all_df)))
    print("[INFO] Runs missing final_forgetting: {}".format(missing_forgetting))

    analysis_df = all_df
    if args.only_valid_forgetting:
        analysis_df = filter_valid_forgetting(all_df)

    summary_df = build_summary_table(analysis_df)
    pretty_df = build_pretty_summary(summary_df)
    pairwise_sig_df = build_pairwise_significance_table(analysis_df)

    print("\n=== SUMMARY TABLE ===")
    print(pretty_df.to_string(index=False))

    if pairwise_sig_df.empty:
        print("\n=== PAIRWISE SIGNIFICANCE ===")
        print("[INFO] No pairwise matched-seed comparisons available.")
    else:
        print("\n=== PAIRWISE SIGNIFICANCE (first rows) ===")
        cols_to_show = [
            "dataset_name",
            "adapter_bottleneck",
            "metric",
            "label_a",
            "label_b",
            "n_pairs_used",
            "mean_a",
            "mean_b",
            "mean_diff_a_minus_b",
            "p_value",
            "significant_p_lt_0_05",
            "fdr_significant_bh_0_05",
            "winner",
        ]
        print(pairwise_sig_df[cols_to_show].head(20).to_string(index=False))

    all_runs_csv, summary_csv, summary_pretty_csv, missing_forgetting_csv, pairwise_sig_csv = save_outputs(
        all_df, summary_df, args.out_dir, pairwise_sig_df=pairwise_sig_df
    )
    save_plot_bar_forgetting(summary_df, args.out_dir)
    save_plot_accuracy_vs_forgetting(summary_df, args.out_dir)
    save_plot_accuracy_vs_fairness(summary_df, args.out_dir)
    save_plot_accuracy_vs_energy(summary_df, args.out_dir)
    save_journal_small_multiples(summary_df, args.out_dir)
    save_journal_heterogeneity_sweep(summary_df, args.out_dir)
    save_journal_bottleneck_sweep(summary_df, args.out_dir)

    print("\n=== CHECKS ===")
    print("Total runs found: {}".format(len(all_df)))
    print("Runs missing final_forgetting: {}".format(missing_forgetting))
    print("Runs used for summary/plots: {}".format(len(analysis_df)))
    print("Pairwise significance rows: {}".format(len(pairwise_sig_df)))

    if not args.only_valid_forgetting and missing_forgetting > 0:
        print("[WARN] Summary includes groups where forgetting may be NaN.")
        print("[WARN] Use --only_valid_forgetting for paper-quality tables.")

    if missing_forgetting > 0:
        print("[WARN] Missing-run list saved to: {}".format(missing_forgetting_csv))

    print("\nSaved outputs:")
    print(" - {}".format(all_runs_csv))
    print(" - {}".format(summary_csv))
    print(" - {}".format(summary_pretty_csv))
    print(" - {}".format(missing_forgetting_csv))
    print(" - {}".format(pairwise_sig_csv))
    print(" - {}/*_forgetting_bar.(png|pdf)".format(args.out_dir))
    print(" - {}/*_accuracy_vs_forgetting.(png|pdf)".format(args.out_dir))
    print(" - {}/*_accuracy_vs_fairness.(png|pdf)".format(args.out_dir))
    print(" - {}/*_accuracy_vs_energy.(png|pdf)".format(args.out_dir))
    print(" - {}/journal_small_multiples_overview.(png|pdf) [if multiple datasets present]".format(args.out_dir))
    print(" - {}/*_heterogeneity_sweep.(png|pdf) [if multiple alpha settings present]".format(args.out_dir))
    print(" - {}/*_bottleneck_sweep.(png|pdf) [if multiple bottlenecks present]".format(args.out_dir))


if __name__ == "__main__":
    main()
