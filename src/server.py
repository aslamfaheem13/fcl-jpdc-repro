#server.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence
import json
import math
import os
import platform
import re
import shutil
import subprocess
import time

import torch
import torch.nn as nn


# ============================================================
# Helpers
# ============================================================

def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.element_size() * t.numel())


def _state_nbytes(state: Dict[str, torch.Tensor]) -> int:
    return sum(_tensor_nbytes(v) for v in state.values() if torch.is_tensor(v))


def _load_partial_state(model: nn.Module, partial: Dict[str, torch.Tensor]) -> None:
    sd = model.state_dict()
    for k, v in partial.items():
        if k not in sd:
            raise KeyError(f"[Server] Attempted to load unknown key: {k}")
        if tuple(sd[k].shape) != tuple(v.shape):
            raise ValueError(
                f"[Server] Shape mismatch for key='{k}': "
                f"model={tuple(sd[k].shape)} incoming={tuple(v.shape)}"
            )
        sd[k] = v.detach().clone()
    model.load_state_dict(sd, strict=False)


def _validate_client_states(states: List[Dict[str, torch.Tensor]]) -> None:
    if not states:
        raise ValueError("[Server] No client states provided for aggregation")

    ref_keys = list(states[0].keys())
    ref_shapes = {k: tuple(v.shape) for k, v in states[0].items()}

    for i, sd in enumerate(states[1:], start=1):
        keys = list(sd.keys())
        if keys != ref_keys:
            missing = sorted(set(ref_keys) - set(keys))
            extra = sorted(set(keys) - set(ref_keys))
            raise ValueError(
                f"[Server] State dict key mismatch for client_state[{i}]. "
                f"Missing={missing[:10]} Extra={extra[:10]}"
            )
        for k in ref_keys:
            if tuple(sd[k].shape) != ref_shapes[k]:
                raise ValueError(
                    f"[Server] Shape mismatch at client_state[{i}] key='{k}': "
                    f"expected={ref_shapes[k]} got={tuple(sd[k].shape)}"
                )


def _weighted_avg_state_dict(
    states: List[Dict[str, torch.Tensor]],
    weights: List[int],
) -> Dict[str, torch.Tensor]:
    if not states:
        raise ValueError("[Server] _weighted_avg_state_dict: empty states")
    if len(states) != len(weights):
        raise ValueError("[Server] states and weights length mismatch")

    _validate_client_states(states)

    total_weight = float(sum(weights))
    if total_weight <= 0:
        raise ValueError("[Server] total_weight must be positive")

    out: Dict[str, torch.Tensor] = {}
    keys = list(states[0].keys())

    for k in keys:
        vals = [sd[k].detach() for sd in states]
        dtype = vals[0].dtype
        device = vals[0].device
        acc = torch.zeros_like(vals[0], dtype=torch.float32, device=device)

        for v, w in zip(vals, weights):
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"[Server] NaN/Inf detected in client update for key='{k}'")
            acc += v.to(torch.float32) * (float(w) / total_weight)

        out[k] = acc.to(dtype)

    return out


def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    return getattr(cfg, name, default)


def _safe_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return bool(default)
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


# ============================================================
# Dataclasses
# ============================================================

@dataclass
class CommStats:
    upload_bytes_total: int = 0
    download_bytes_total: int = 0
    upload_keys_total: int = 0
    download_keys_total: int = 0

    def add_upload(self, nbytes: int, nkeys: int) -> None:
        self.upload_bytes_total += int(nbytes)
        self.upload_keys_total += int(nkeys)

    def add_download(self, nbytes: int, nkeys: int) -> None:
        self.download_bytes_total += int(nbytes)
        self.download_keys_total += int(nkeys)


@dataclass
class EvalResult:
    task_id: int
    num_samples: int
    accuracy: float


@dataclass
class JetsonPowerStats:
    enabled: bool = False
    tegrastats_found: bool = False
    logfile_path: Optional[str] = None
    sample_count: int = 0
    avg_power_w: float = float("nan")
    peak_power_w: float = float("nan")
    energy_j: float = float("nan")
    avg_cpu_util: float = float("nan")
    peak_cpu_util: float = float("nan")
    avg_gpu_util: float = float("nan")
    peak_gpu_util: float = float("nan")
    avg_ram_mb: float = float("nan")
    peak_ram_mb: float = float("nan")
    avg_swap_mb: float = float("nan")
    peak_swap_mb: float = float("nan")
    power_source: str = "NA"


# ============================================================
# Jetson tegrastats helpers
# ============================================================

_RAM_RE = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_SWAP_RE = re.compile(r"SWAP\s+(\d+)/(\d+)MB")
_GR3D_RE = re.compile(r"GR3D_FREQ\s+(\d+)%")
_VDD_RE = re.compile(r"(VDD_IN|POM_5V_IN)\s+(\d+)(mW|W)")
_CPU_BRACKET_RE = re.compile(r"CPU\s+\[([^\]]+)\]")


def _parse_tegrastats_line(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}

    m = _RAM_RE.search(line)
    if m:
        out["ram_used_mb"] = float(m.group(1))

    m = _SWAP_RE.search(line)
    if m:
        out["swap_used_mb"] = float(m.group(1))

    m = _GR3D_RE.search(line)
    if m:
        out["gpu_util"] = float(m.group(1))

    m = _VDD_RE.search(line)
    if m:
        source = str(m.group(1))
        value = float(m.group(2))
        unit = str(m.group(3)).lower()
        power_w = value / 1000.0 if unit == "mw" else value
        out["power_w"] = power_w
        out["power_source"] = source

    m = _CPU_BRACKET_RE.search(line)
    if m:
        entries = [x.strip() for x in m.group(1).split(",") if x.strip()]
        cpu_vals = []
        for item in entries:
            m2 = re.search(r"(\d+)%", item)
            if m2:
                cpu_vals.append(float(m2.group(1)))
        if cpu_vals:
            out["cpu_util"] = float(sum(cpu_vals) / len(cpu_vals))

    return out


class _TegrastatsMonitor:
    def __init__(self, out_path: str, interval_ms: int = 500):
        self.out_path = out_path
        self.interval_ms = max(100, int(interval_ms))
        self.proc: Optional[subprocess.Popen] = None
        self._fh = None
        self.available = shutil.which("tegrastats") is not None

    def start(self) -> None:
        if not self.available:
            return
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        self._fh = open(self.out_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=self._fh,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def stop(self) -> None:
        try:
            if self.proc is not None and self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    self.proc.kill()
        finally:
            self.proc = None
            if self._fh is not None:
                try:
                    self._fh.flush()
                except Exception:
                    pass
                self._fh.close()
                self._fh = None

    def summarize(self, wall_time_sec: float) -> JetsonPowerStats:
        stats = JetsonPowerStats(
            enabled=True,
            tegrastats_found=self.available,
            logfile_path=self.out_path,
        )

        if not self.available or not os.path.exists(self.out_path):
            return stats

        power_vals: List[float] = []
        cpu_vals: List[float] = []
        gpu_vals: List[float] = []
        ram_vals: List[float] = []
        swap_vals: List[float] = []
        power_source = "NA"

        with open(self.out_path, "r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parsed = _parse_tegrastats_line(line)
                if not parsed:
                    continue
                if "power_w" in parsed:
                    power_vals.append(float(parsed["power_w"]))
                    power_source = str(parsed.get("power_source", power_source))
                if "cpu_util" in parsed:
                    cpu_vals.append(float(parsed["cpu_util"]))
                if "gpu_util" in parsed:
                    gpu_vals.append(float(parsed["gpu_util"]))
                if "ram_used_mb" in parsed:
                    ram_vals.append(float(parsed["ram_used_mb"]))
                if "swap_used_mb" in parsed:
                    swap_vals.append(float(parsed["swap_used_mb"]))

        stats.sample_count = max(
            len(power_vals), len(cpu_vals), len(gpu_vals), len(ram_vals), len(swap_vals)
        )
        stats.power_source = power_source

        if power_vals:
            stats.avg_power_w = float(sum(power_vals) / len(power_vals))
            stats.peak_power_w = float(max(power_vals))
            stats.energy_j = float(stats.avg_power_w * max(0.0, float(wall_time_sec)))
        if cpu_vals:
            stats.avg_cpu_util = float(sum(cpu_vals) / len(cpu_vals))
            stats.peak_cpu_util = float(max(cpu_vals))
        if gpu_vals:
            stats.avg_gpu_util = float(sum(gpu_vals) / len(gpu_vals))
            stats.peak_gpu_util = float(max(gpu_vals))
        if ram_vals:
            stats.avg_ram_mb = float(sum(ram_vals) / len(ram_vals))
            stats.peak_ram_mb = float(max(ram_vals))
        if swap_vals:
            stats.avg_swap_mb = float(sum(swap_vals) / len(swap_vals))
            stats.peak_swap_mb = float(max(swap_vals))

        return stats


# ============================================================
# Server
# ============================================================

class Server:
    def __init__(
        self,
        model: nn.Module,
        clients: List[Any],
        cfg: Any,
    ):
        if model is None:
            raise TypeError("Server requires a non-null 'model'")
        if not isinstance(model, nn.Module):
            raise TypeError(f"Server expected nn.Module for model, got {type(model)}")
        if clients is None or len(clients) == 0:
            raise TypeError("Server requires a non-empty 'clients' list")
        if cfg is None:
            raise TypeError("Server requires a non-null 'cfg'")

        self.model = model
        self.clients = list(clients)
        self.cfg = cfg
        self.method = str(cfg.method).upper()
        self.fl_algo = str(_cfg_get(cfg, "fl_algo", "FEDAVG")).upper()
        self.comm = CommStats()

        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

        self.metrics_path = os.path.join(self.cfg.out_dir, "metrics.jsonl")
        self.summary_path = os.path.join(self.cfg.out_dir, "summary.json")
        self.manifest_path = os.path.join(self.cfg.out_dir, "manifest.json")
        self.final_model_path = os.path.join(self.cfg.out_dir, "final_model.pt")
        self.task_matrix_path = os.path.join(self.cfg.out_dir, "task_accuracy_matrix.json")
        self.tegrastats_log_path = os.path.join(self.cfg.out_dir, "tegrastats.log")

        os.makedirs(self.cfg.out_dir, exist_ok=True)

        self.server_control: Dict[str, torch.Tensor] = {}
        self._test_loader_cache: Dict[int, Any] = {}

        self.num_tasks = int(self.cfg.num_tasks)
        self.task_accuracy_matrix: List[List[float]] = [
            [float("nan")] * self.num_tasks for _ in range(self.num_tasks)
        ]

        self.forward_transfer_records: List[Dict[str, float]] = []

    # ---------------- Config metadata helpers ----------------

    def _alpha_value(self) -> float:
        for key in ["alpha", "dirichlet_alpha"]:
            v = _cfg_get(self.cfg, key, None)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return float("nan")

    def _replay_enabled(self) -> bool:
        replay_flag = _cfg_get(self.cfg, "replay", None)
        if replay_flag is not None:
            return bool(replay_flag)

        rpc = _cfg_get(self.cfg, "replay_per_class", 0)
        try:
            return int(rpc) > 0
        except Exception:
            return False

    def _trainable_scope(self) -> str:
        mapping = {
            "FULL_FEDAVG": "head_only",
            "TRUE_FEDAVG": "full_model_no_adapters",
            "SHARED_ADAPTER": "shared_adapter_plus_heads",
            "TASK_ADAPTER": "task_adapter_plus_heads",
            "LWF": "full_model_no_adapters",
            "EWC": "full_model_no_adapters",
            "MAS": "full_model_no_adapters",
            "SI": "full_model_no_adapters",
        }
        return mapping.get(self.method, "unknown")

    def _backbone_frozen(self) -> bool:
        try:
            params = list(self.model.backbone.parameters())
            if not params:
                return False
            return not any(bool(p.requires_grad) for p in params)
        except Exception:
            return self.method == "FULL_FEDAVG"

    def _backbone_pretrained(self) -> bool:
        for key in ("pretrained_backbone", "backbone_pretrained", "pretrained"):
            v = _cfg_get(self.cfg, key, None)
            if v is not None:
                return _safe_bool(v, default=False)
        v = getattr(self.model, "pretrained_backbone", None)
        if v is not None:
            return _safe_bool(v, default=False)
        return False

    def _task_id_known_at_inference(self) -> bool:
        return True

    def _evaluation_protocol(self) -> str:
        ds = str(_cfg_get(self.cfg, "dataset_name", "")).lower()
        if ds == "digit5":
            return "domain_incremental_task_aware"
        return "class_incremental_task_aware"

    def _common_run_metadata(self) -> Dict[str, Any]:
        return {
            "dataset_name": str(_cfg_get(self.cfg, "dataset_name", "UNKNOWN")),
            "method": self.method,
            "fl_algo": self.fl_algo,
            "seed": int(_cfg_get(self.cfg, "seed", -1)),
            "alpha": self._alpha_value(),
            "replay": bool(self._replay_enabled()),
            "replay_per_class": int(_cfg_get(self.cfg, "replay_per_class", 0)),
            "replay_lambda": float(_cfg_get(self.cfg, "replay_lambda", 0.0)),
            "mu": float(_cfg_get(self.cfg, "mu", 0.0)),
            "tasks": int(_cfg_get(self.cfg, "num_tasks", 0)),
            "rounds_per_task": int(_cfg_get(self.cfg, "rounds_per_task", 0)),
            "local_epochs": int(_cfg_get(self.cfg, "local_epochs", 0)),
            "head_train_mode": str(_cfg_get(self.cfg, "head_train_mode", "current")),
            "num_clients": int(_cfg_get(self.cfg, "num_clients", len(self.clients))),
            "adapter_bottleneck": int(
                _cfg_get(self.cfg, "adapter_bottleneck", getattr(self.model, "adapter_bottleneck", 16))
            ),
            "backbone_pretrained": bool(self._backbone_pretrained()),
            "backbone_frozen": bool(self._backbone_frozen()),
            "trainable_scope": self._trainable_scope(),
            "evaluation_protocol": self._evaluation_protocol(),
            "task_id_known_at_inference": bool(self._task_id_known_at_inference()),
            "enable_power": bool(_safe_bool(_cfg_get(self.cfg, "enable_power", False))),
            "tegrastats_interval_ms": int(_cfg_get(self.cfg, "tegrastats_interval_ms", 500)),
        }

    def _is_local_only(self) -> bool:
        return self.fl_algo == "LOCAL_ONLY"

    def _is_joint_only(self) -> bool:
        return self.fl_algo == "JOINT_ONLY"

    # ---------------- Naming rules ----------------

    def _head_rule(self, task_id: int):
        mode = str(getattr(self.cfg, "head_train_mode", "current")).lower()
        if mode not in ("current", "seen"):
            mode = "current"

        if mode == "current":
            prefix = f"classifiers.{int(task_id)}."
            return lambda k: k.startswith(prefix)

        def rule(k: str) -> bool:
            if not k.startswith("classifiers."):
                return False
            parts = k.split(".")
            if len(parts) < 3:
                return False
            try:
                t = int(parts[1])
            except Exception:
                return False
            return t <= int(task_id)

        return rule

    def _select_sync_keys(self, task_id: int) -> List[str]:
        sd = self.model.state_dict()
        keys = list(sd.keys())
        head_rule = self._head_rule(task_id)
        m = self.method

        if m == "FULL_FEDAVG":
            return [k for k in keys if head_rule(k)]

        if m in ("TRUE_FEDAVG", "LWF", "EWC", "MAS", "SI"):
            out = []
            for k in keys:
                if "shared_adapter" in k:
                    continue
                if "task_adapters" in k:
                    continue
                if k.startswith("classifiers."):
                    if head_rule(k):
                        out.append(k)
                else:
                    out.append(k)
            return out

        if m == "SHARED_ADAPTER":
            out = []
            for k in keys:
                if "shared_adapter" in k:
                    out.append(k)
                elif k.startswith("classifiers.") and head_rule(k):
                    out.append(k)
            return out

        if m == "TASK_ADAPTER":
            out = []
            task_prefix = f"task_adapters.{int(task_id)}."
            for k in keys:
                if k.startswith(task_prefix):
                    out.append(k)
                elif k.startswith("classifiers.") and head_rule(k):
                    out.append(k)
            return out

        raise ValueError(f"[Server] Unknown method: {self.method}")

    def _get_partial_state(self, keys: Sequence[str]) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        partial: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k not in sd:
                raise KeyError(f"[Server] Missing model state key: {k}")
            partial[k] = sd[k].detach().clone()
        return partial

    def _sync_all_clients_from_server(self, task_id: int) -> Dict[str, torch.Tensor]:
        keys = self._select_sync_keys(task_id)
        if len(keys) == 0:
            raise RuntimeError(f"[Server] No sync keys selected for method={self.method} task_id={task_id}")

        payload = self._get_partial_state(keys)
        for c in self.clients:
            c.load_global_state(payload)
        return payload

    # ---------------- SCAFFOLD helpers ----------------

    def _init_server_control_if_needed(self, keys: List[str]) -> None:
        if self.server_control:
            return

        sd = self.model.state_dict()
        self.server_control = {}
        for k in keys:
            if k not in sd:
                raise KeyError(f"[Server] Missing key for server control init: {k}")
            self.server_control[k] = torch.zeros_like(
                sd[k],
                dtype=torch.float32,
                device=self.device,
            )

    def _update_server_control(self, updates: List[Any], sync_keys: List[str]) -> None:
        if self.fl_algo != "SCAFFOLD":
            return
        if not updates:
            return

        valid_updates = [u for u in updates if getattr(u, "client_control_delta", None) is not None]
        if not valid_updates:
            return

        for k in sync_keys:
            deltas = []
            for u in valid_updates:
                if k in u.client_control_delta:
                    deltas.append(u.client_control_delta[k].to(self.device, dtype=torch.float32))

            if not deltas:
                continue

            avg_delta = torch.zeros_like(deltas[0], dtype=torch.float32, device=self.device)
            for d in deltas:
                avg_delta += d
            avg_delta /= float(len(deltas))

            self.server_control[k] = self.server_control[k] + avg_delta

    # ---------------- Communication ----------------

    def broadcast(self, task_id: int, selected_clients: List[Any]) -> Dict[str, torch.Tensor]:
        keys = self._select_sync_keys(task_id)
        if len(keys) == 0:
            raise RuntimeError(f"[Server] No sync keys selected for method={self.method} task_id={task_id}")

        payload = self._get_partial_state(keys)
        nbytes = _state_nbytes(payload)
        nkeys = len(payload)

        if self.fl_algo == "SCAFFOLD":
            self._init_server_control_if_needed(keys)

        for c in selected_clients:
            c.load_global_state(payload)
            if self.fl_algo == "SCAFFOLD":
                c.load_server_control(self.server_control)
            self.comm.add_download(nbytes=nbytes, nkeys=nkeys)

        return payload

    def aggregate(self, updates: List[Any], task_id: int) -> Dict[str, torch.Tensor]:
        if not updates:
            raise ValueError("[Server] Cannot aggregate empty updates")

        states = [u.state_dict for u in updates]
        weights = [int(u.num_samples) for u in updates]

        if min(weights) <= 0:
            raise ValueError(f"[Server] Invalid aggregation weights: {weights}")

        avg = _weighted_avg_state_dict(states, weights)
        _load_partial_state(self.model, avg)
        return avg

    def _run_joint_only_round(self, task_id: int, selected_clients: List[Any]) -> Dict[str, Any]:
        sync_keys = self._select_sync_keys(task_id)
        if len(sync_keys) == 0:
            raise RuntimeError(f"[Server] No sync keys selected for method={self.method} task_id={task_id}")

        local_losses: List[float] = []
        local_num_samples: List[int] = []
        replay_memory_values: List[float] = []
        last_update = None

        for client in selected_clients:
            payload = self._get_partial_state(sync_keys)
            client.load_global_state(payload)

            update = client.train(
                task_id=task_id,
                epochs=int(self.cfg.local_epochs),
                sync_keys=sync_keys,
            )
            last_update = update

            _load_partial_state(self.model, update.state_dict)

            local_losses.append(float(update.metrics.get("train_loss", 0.0)))
            local_num_samples.append(int(update.num_samples))

            if "replay_memory_mb" in update.metrics:
                try:
                    replay_memory_values.append(float(update.metrics["replay_memory_mb"]))
                except Exception:
                    pass

        self._sync_all_clients_from_server(task_id=task_id)

        payload_nbytes = 0
        if last_update is not None:
            try:
                payload_nbytes = int(_state_nbytes(last_update.state_dict))
            except Exception:
                payload_nbytes = 0

        return {
            "sync_keys": sync_keys,
            "payload_nbytes": payload_nbytes,
            "local_losses": local_losses,
            "local_num_samples": local_num_samples,
            "replay_memory_values": replay_memory_values,
        }

    # ---------------- Evaluation ----------------

    def _adapter_mode(self) -> str:
        if self.method in ("FULL_FEDAVG", "TRUE_FEDAVG", "LWF", "EWC", "MAS", "SI"):
            return "none"
        if self.method == "SHARED_ADAPTER":
            return "shared"
        if self.method == "TASK_ADAPTER":
            return "task"
        raise ValueError(f"[Server] Unknown method: {self.method}")

    def _get_cached_test_loader(self, task_id: int):
        if task_id not in self._test_loader_cache:
            try:
                from .data import get_task_test_loader  # type: ignore
            except Exception:
                from src.data import get_task_test_loader  # type: ignore

            loader, task_classes = get_task_test_loader(
                dataset_name=self.cfg.dataset_name,
                data_root=self.cfg.data_root,
                task_id=task_id,
                classes_per_task=self.cfg.classes_per_task,
                batch_size=128,
            )
            self._test_loader_cache[task_id] = (loader, task_classes)
        return self._test_loader_cache[task_id]

    def _chance_accuracy_for_task(self, task_id: int) -> float:
        _loader, task_classes = self._get_cached_test_loader(task_id)
        n_classes = len(task_classes) if task_classes is not None else int(self.cfg.classes_per_task)
        if n_classes <= 0:
            return 0.0
        return 1.0 / float(n_classes)

    def evaluate_task_with_model(self, model: nn.Module, task_id: int) -> EvalResult:
        loader, _task_classes = self._get_cached_test_loader(task_id)

        model.eval()
        adapter_mode = self._adapter_mode()

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                logits = model(x, task_id=task_id, adapter_mode=adapter_mode)
                pred = logits.argmax(dim=1)

                # Convert global class labels to task-local labels
                if str(self.cfg.dataset_name).lower() in ("cifar10", "cifar100", "tinyimagenet"):
                    y_eval = y - int(task_id) * int(self.cfg.classes_per_task)
                else:
                    y_eval = y

                correct += int((pred == y_eval).sum().item())
                total += int(y.numel())
        acc = float(correct / total) if total > 0 else 0.0
        return EvalResult(task_id=task_id, num_samples=total, accuracy=acc)

    def evaluate_task(self, task_id: int) -> EvalResult:
        return self.evaluate_task_with_model(self.model, task_id)

    def evaluate_seen_tasks(self, current_task_id: int) -> Dict[str, Any]:
        per_task: List[Dict[str, Any]] = []
        for task_id in range(current_task_id + 1):
            res = self.evaluate_task(task_id)
            per_task.append(asdict(res))

        avg_acc = float(sum(x["accuracy"] for x in per_task) / len(per_task)) if per_task else 0.0
        return {
            "eval_seen_avg_accuracy": avg_acc,
            "eval_seen_tasks": per_task,
        }

    def evaluate_seen_tasks_local_only(self, current_task_id: int) -> Dict[str, Any]:
        per_task: List[Dict[str, Any]] = []

        for task_id in range(current_task_id + 1):
            client_task_accs: List[float] = []
            num_samples_ref: Optional[int] = None

            for client in self.clients:
                res = self.evaluate_task_with_model(client.model, task_id)
                client_task_accs.append(float(res.accuracy))
                if num_samples_ref is None:
                    num_samples_ref = int(res.num_samples)

            avg_acc = float(sum(client_task_accs) / len(client_task_accs)) if client_task_accs else 0.0
            per_task.append({
                "task_id": int(task_id),
                "num_samples": int(num_samples_ref or 0),
                "accuracy": avg_acc,
            })

        avg_seen = float(sum(x["accuracy"] for x in per_task) / len(per_task)) if per_task else 0.0
        return {
            "eval_seen_avg_accuracy": avg_seen,
            "eval_seen_tasks": per_task,
        }

    def evaluate_forward_task_local_only(self, task_id: int) -> float:
        accs: List[float] = []
        for client in self.clients:
            res = self.evaluate_task_with_model(client.model, task_id)
            accs.append(float(res.accuracy))
        return float(sum(accs) / len(accs)) if accs else 0.0

    def evaluate_client_fairness(self, current_task_id: int, selected_clients: List[Any]) -> Dict[str, Any]:
        client_accs: List[float] = []
        client_details: List[Dict[str, float]] = []

        for client in selected_clients:
            out = client.evaluate_seen_local(current_task_id=current_task_id)
            acc = float(out["seen_local_accuracy"])
            client_accs.append(acc)
            client_details.append({
                "client_id": float(out["client_id"]),
                "seen_local_accuracy": acc,
                "seen_local_num_samples": float(out["seen_local_num_samples"]),
            })

        if not client_accs:
            return {
                "client_seen_accuracy_mean": 0.0,
                "client_seen_accuracy_std": 0.0,
                "client_seen_accuracy_min": 0.0,
                "client_seen_accuracy_max": 0.0,
                "client_seen_accuracy_per_client": [],
            }

        mean_acc = float(sum(client_accs) / len(client_accs))

        if len(client_accs) > 1:
            var = sum((x - mean_acc) ** 2 for x in client_accs) / len(client_accs)
            std_acc = float(var ** 0.5)
        else:
            std_acc = 0.0

        return {
            "client_seen_accuracy_mean": mean_acc,
            "client_seen_accuracy_std": std_acc,
            "client_seen_accuracy_min": float(min(client_accs)),
            "client_seen_accuracy_max": float(max(client_accs)),
            "client_seen_accuracy_per_client": client_details,
        }

    # ---------------- Continual-learning metrics ----------------

    def _update_task_accuracy_matrix(self, trained_task_id: int, eval_seen_tasks: List[Dict[str, Any]]) -> None:
        for item in eval_seen_tasks:
            eval_task_id = int(item["task_id"])
            acc = float(item["accuracy"])
            self.task_accuracy_matrix[trained_task_id][eval_task_id] = acc

    def _maybe_record_forward_transfer(self, trained_task_id: int) -> Optional[Dict[str, float]]:
        next_task_id = int(trained_task_id) + 1
        if next_task_id >= self.num_tasks:
            return None

        if self._is_local_only():
            acc_before = self.evaluate_forward_task_local_only(next_task_id)
        else:
            acc_before = float(self.evaluate_task(next_task_id).accuracy)

        chance_acc = self._chance_accuracy_for_task(next_task_id)
        fwt_val = float(acc_before - chance_acc)

        rec = {
            "source_task_id": float(trained_task_id),
            "target_task_id": float(next_task_id),
            "accuracy_before_training": float(acc_before),
            "chance_accuracy": float(chance_acc),
            "fwt": float(fwt_val),
        }
        self.forward_transfer_records.append(rec)
        return rec

    def _compute_forward_transfer_summary(self) -> Dict[str, Any]:
        if not self.forward_transfer_records:
            return {
                "avg_fwt": 0.0,
                "fwt_per_task": [],
            }

        vals = [float(x["fwt"]) for x in self.forward_transfer_records]
        avg_fwt = float(sum(vals) / len(vals)) if vals else 0.0

        fwt_per_task: List[Dict[str, float]] = []
        for x in self.forward_transfer_records:
            fwt_per_task.append({
                "source_task_id": float(x["source_task_id"]),
                "target_task_id": float(x["target_task_id"]),
                "accuracy_before_training": float(x["accuracy_before_training"]),
                "chance_accuracy": float(x["chance_accuracy"]),
                "fwt": float(x["fwt"]),
            })

        return {
            "avg_fwt": avg_fwt,
            "fwt_per_task": fwt_per_task,
        }

    def _safe_float_matrix(self) -> List[List[Any]]:
        out: List[List[Any]] = []
        for row in self.task_accuracy_matrix:
            clean_row = []
            for v in row:
                if isinstance(v, float) and math.isnan(v):
                    clean_row.append(None)
                else:
                    clean_row.append(float(v))
            out.append(clean_row)
        return out

    def _compute_forgetting_from_matrix(self, upto_task_id: Optional[int] = None) -> Dict[str, Any]:
        if upto_task_id is None:
            upto_task_id = self.num_tasks - 1

        T = int(upto_task_id)
        if T < 0:
            return {
                "avg_forgetting": 0.0,
                "forgetting_per_task": [],
                "final_avg_accuracy": 0.0,
                "avg_bwt": 0.0,
                "bwt_per_task": [],
                "avg_accuracy_drop_from_peak": 0.0,
                "accuracy_drop_from_peak_per_task": [],
            }

        final_accs = []
        for k in range(T + 1):
            v = self.task_accuracy_matrix[T][k]
            if not (isinstance(v, float) and math.isnan(v)):
                final_accs.append(float(v))
        final_avg_accuracy = float(sum(final_accs) / len(final_accs)) if final_accs else 0.0

        forgetting_per_task: List[Dict[str, float]] = []
        forgetting_vals: List[float] = []
        bwt_per_task: List[Dict[str, float]] = []
        bwt_vals: List[float] = []
        accuracy_drop_from_peak_per_task: List[Dict[str, float]] = []
        peak_drop_vals: List[float] = []

        for k in range(T + 1):
            values = []
            for t in range(k, T + 1):
                v = self.task_accuracy_matrix[t][k]
                if not (isinstance(v, float) and math.isnan(v)):
                    values.append(float(v))

            if not values:
                continue

            initial_v = self.task_accuracy_matrix[k][k]
            final_v = self.task_accuracy_matrix[T][k]
            best_v = max(values)

            if isinstance(initial_v, float) and math.isnan(initial_v):
                continue
            if isinstance(final_v, float) and math.isnan(final_v):
                continue

            initial_v = float(initial_v)
            final_v = float(final_v)
            best_v = float(best_v)

            drop_k = best_v - final_v
            accuracy_drop_from_peak_per_task.append({
                "task_id": int(k),
                "best_accuracy": best_v,
                "final_accuracy": final_v,
                "accuracy_drop_from_peak": float(drop_k),
            })
            peak_drop_vals.append(float(drop_k))

            bwt_k = final_v - initial_v
            bwt_per_task.append({
                "task_id": int(k),
                "initial_accuracy": initial_v,
                "final_accuracy": final_v,
                "bwt": float(bwt_k),
            })
            bwt_vals.append(float(bwt_k))

            if k < T:
                f_k = best_v - final_v
                forgetting_per_task.append({
                    "task_id": int(k),
                    "best_accuracy": best_v,
                    "final_accuracy": final_v,
                    "forgetting": float(f_k),
                })
                forgetting_vals.append(float(f_k))

        avg_forgetting = float(sum(forgetting_vals) / len(forgetting_vals)) if forgetting_vals else 0.0
        avg_bwt = float(sum(bwt_vals) / len(bwt_vals)) if bwt_vals else 0.0
        avg_accuracy_drop_from_peak = float(sum(peak_drop_vals) / len(peak_drop_vals)) if peak_drop_vals else 0.0

        return {
            "avg_forgetting": avg_forgetting,
            "forgetting_per_task": forgetting_per_task,
            "final_avg_accuracy": final_avg_accuracy,
            "avg_bwt": avg_bwt,
            "bwt_per_task": bwt_per_task,
            "avg_accuracy_drop_from_peak": avg_accuracy_drop_from_peak,
            "accuracy_drop_from_peak_per_task": accuracy_drop_from_peak_per_task,
        }

    def _save_task_accuracy_matrix(self) -> None:
        payload = {
            "num_tasks": self.num_tasks,
            "matrix": self._safe_float_matrix(),
        }
        with open(self.task_matrix_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    # ---------------- Logging / saving ----------------

    def _append_metrics(self, record: Dict[str, Any]) -> None:
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _safe_git_commit(self) -> str:
        try:
            out = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            return out
        except Exception:
            return "unknown"

    def _write_manifest(self) -> None:
        try:
            cfg_dict = asdict(self.cfg)
        except Exception:
            cfg_dict = dict(vars(self.cfg))

        manifest = {
            "timestamp_unix": time.time(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": self._safe_git_commit(),
            "config": cfg_dict,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    def _save_final_model(self) -> None:
        if self._is_local_only():
            torch.save(
                {
                    "client_states": [c.model.state_dict() for c in self.clients],
                    "num_clients": len(self.clients),
                },
                self.final_model_path,
            )
        else:
            torch.save(self.model.state_dict(), self.final_model_path)

        if getattr(self.cfg, "backbone_ckpt_out", None) and not self._is_local_only():
            self.model.save_backbone(self.cfg.backbone_ckpt_out)

    # ---------------- Client selection ----------------

    def select_clients(self, task_id: int, round_id: int) -> List[Any]:
        return self.clients

    # ---------------- Main training loop ----------------

    def run(self) -> Dict[str, Any]:
        self._write_manifest()

        total_rounds = 0
        wall_start = time.time()
        last_fairness_summary = {
            "client_seen_accuracy_mean": 0.0,
            "client_seen_accuracy_std": 0.0,
            "client_seen_accuracy_min": 0.0,
            "client_seen_accuracy_max": 0.0,
            "client_seen_accuracy_per_client": [],
        }

        power_monitor: Optional[_TegrastatsMonitor] = None
        if _safe_bool(_cfg_get(self.cfg, "enable_power", False), default=False):
            power_monitor = _TegrastatsMonitor(
                out_path=self.tegrastats_log_path,
                interval_ms=int(_cfg_get(self.cfg, "tegrastats_interval_ms", 500)),
            )
            try:
                power_monitor.start()
            except Exception as e:
                print(f"[WARN] Failed to start tegrastats monitor: {e}")
                power_monitor = None

        try:
            for task_id in range(self.num_tasks):
                for round_id in range(int(self.cfg.rounds_per_task)):
                    round_start = time.time()

                    selected_clients = self.select_clients(task_id=task_id, round_id=round_id)
                    if len(selected_clients) == 0:
                        raise RuntimeError(f"[Server] No selected clients for task={task_id} round={round_id}")

                    if self._is_joint_only():
                        joint_stats = self._run_joint_only_round(
                            task_id=task_id,
                            selected_clients=selected_clients,
                        )
                        sync_keys = list(joint_stats["sync_keys"])
                        payload = {}
                        local_losses = list(joint_stats["local_losses"])
                        local_num_samples = list(joint_stats["local_num_samples"])
                        replay_memory_values = list(joint_stats["replay_memory_values"])
                        payload_nbytes = int(joint_stats["payload_nbytes"])

                    else:
                        if self._is_local_only():
                            sync_keys = []
                            payload = {}
                        else:
                            payload = self.broadcast(task_id=task_id, selected_clients=selected_clients)
                            sync_keys = list(payload.keys())

                        local_losses = []
                        local_num_samples = []
                        replay_memory_values = []
                        updates = []

                        for client in selected_clients:
                            update = client.train(
                                task_id=task_id,
                                epochs=int(self.cfg.local_epochs),
                                sync_keys=sync_keys,
                            )
                            updates.append(update)
                            local_losses.append(float(update.metrics.get("train_loss", 0.0)))
                            local_num_samples.append(int(update.num_samples))

                            if "replay_memory_mb" in update.metrics:
                                try:
                                    replay_memory_values.append(float(update.metrics["replay_memory_mb"]))
                                except Exception:
                                    pass

                            if not self._is_local_only():
                                self.comm.add_upload(
                                    nbytes=int(update.communication_bytes),
                                    nkeys=len(update.state_dict),
                                )

                        if not self._is_local_only():
                            self.aggregate(updates=updates, task_id=task_id)

                            if self.fl_algo == "SCAFFOLD":
                                self._update_server_control(updates=updates, sync_keys=sync_keys)

                        payload_nbytes = 0 if self._is_local_only() else int(_state_nbytes(payload))

                    if self._is_local_only():
                        eval_summary = self.evaluate_seen_tasks_local_only(current_task_id=task_id)
                    else:
                        eval_summary = self.evaluate_seen_tasks(current_task_id=task_id)

                    fairness_summary = self.evaluate_client_fairness(
                        current_task_id=task_id,
                        selected_clients=selected_clients,
                    )
                    last_fairness_summary = fairness_summary

                    end_of_task = round_id == int(self.cfg.rounds_per_task) - 1
                    task_level_metrics = None
                    fwt_record = None
                    if end_of_task:
                        self._update_task_accuracy_matrix(
                            trained_task_id=task_id,
                            eval_seen_tasks=eval_summary["eval_seen_tasks"],
                        )
                        task_level_metrics = self._compute_forgetting_from_matrix(upto_task_id=task_id)
                        self._save_task_accuracy_matrix()
                        fwt_record = self._maybe_record_forward_transfer(trained_task_id=task_id)

                    round_time_sec = float(time.time() - round_start)
                    avg_local_loss = float(sum(local_losses) / len(local_losses)) if local_losses else 0.0
                    total_local_samples = int(sum(local_num_samples))
                    avg_replay_memory_mb = (
                        float(sum(replay_memory_values) / len(replay_memory_values))
                        if replay_memory_values else 0.0
                    )

                    record = self._common_run_metadata()
                    record.update({
                        "task_id": int(task_id),
                        "round_id": int(round_id),
                        "num_selected_clients": int(len(selected_clients)),
                        "sync_key_count": int(len(sync_keys)),
                        "sync_payload_bytes_one_way_per_client": int(payload_nbytes),
                        "round_upload_bytes_cumulative": int(self.comm.upload_bytes_total),
                        "round_download_bytes_cumulative": int(self.comm.download_bytes_total),
                        "round_comm_bytes_cumulative": int(self.comm.upload_bytes_total + self.comm.download_bytes_total),
                        "avg_local_train_loss": avg_local_loss,
                        "total_local_samples": total_local_samples,
                        "round_time_sec": round_time_sec,
                        "eval_seen_avg_accuracy": float(eval_summary["eval_seen_avg_accuracy"]),
                        "eval_seen_tasks": eval_summary["eval_seen_tasks"],
                        "client_seen_accuracy_mean": float(fairness_summary["client_seen_accuracy_mean"]),
                        "client_seen_accuracy_std": float(fairness_summary["client_seen_accuracy_std"]),
                        "client_seen_accuracy_min": float(fairness_summary["client_seen_accuracy_min"]),
                        "client_seen_accuracy_max": float(fairness_summary["client_seen_accuracy_max"]),
                        "client_seen_accuracy_per_client": fairness_summary["client_seen_accuracy_per_client"],
                        "avg_replay_memory_mb": avg_replay_memory_mb,
                    })

                    if task_level_metrics is not None:
                        record["task_level_avg_accuracy"] = float(task_level_metrics["final_avg_accuracy"])
                        record["task_level_avg_forgetting"] = float(task_level_metrics["avg_forgetting"])
                        record["task_level_forgetting_per_task"] = task_level_metrics["forgetting_per_task"]
                        record["task_level_avg_bwt"] = float(task_level_metrics["avg_bwt"])
                        record["task_level_bwt_per_task"] = task_level_metrics["bwt_per_task"]
                        record["task_level_avg_accuracy_drop_from_peak"] = float(
                            task_level_metrics["avg_accuracy_drop_from_peak"]
                        )
                        record["task_level_accuracy_drop_from_peak_per_task"] = (
                            task_level_metrics["accuracy_drop_from_peak_per_task"]
                        )

                    if fwt_record is not None:
                        record["forward_transfer_next_task"] = {
                            "source_task_id": float(fwt_record["source_task_id"]),
                            "target_task_id": float(fwt_record["target_task_id"]),
                            "accuracy_before_training": float(fwt_record["accuracy_before_training"]),
                            "chance_accuracy": float(fwt_record["chance_accuracy"]),
                            "fwt": float(fwt_record["fwt"]),
                        }

                    self._append_metrics(record)
                    total_rounds += 1

                    msg = (
                        f"[Server] task={task_id} round={round_id} "
                        f"avg_loss={avg_local_loss:.4f} "
                        f"seen_avg_acc={record['eval_seen_avg_accuracy']:.4f} "
                        f"client_mean={record['client_seen_accuracy_mean']:.4f} "
                        f"client_std={record['client_seen_accuracy_std']:.4f} "
                        f"comm_total={record['round_comm_bytes_cumulative']}"
                    )
                    if task_level_metrics is not None:
                        msg += (
                            f" task_forgetting={task_level_metrics['avg_forgetting']:.4f}"
                            f" task_bwt={task_level_metrics['avg_bwt']:.4f}"
                            f" peak_drop={task_level_metrics['avg_accuracy_drop_from_peak']:.4f}"
                        )
                    if fwt_record is not None:
                        msg += f" next_fwt={float(fwt_record['fwt']):.4f}"
                    print(msg)
        finally:
            if power_monitor is not None:
                power_monitor.stop()

        wall_time_sec = float(time.time() - wall_start)
        self._save_final_model()
        self._save_task_accuracy_matrix()

        final_cl_metrics = self._compute_forgetting_from_matrix(upto_task_id=self.num_tasks - 1)
        final_fwt_metrics = self._compute_forward_transfer_summary()

        power_summary = JetsonPowerStats(enabled=False)
        if power_monitor is not None:
            try:
                power_summary = power_monitor.summarize(wall_time_sec=wall_time_sec)
            except Exception as e:
                print(f"[WARN] Failed to summarize tegrastats: {e}")

        summary = self._common_run_metadata()
        summary.update({
            "total_rounds": int(total_rounds),
            "upload_bytes_total_run": int(self.comm.upload_bytes_total),
            "download_bytes_total_run": int(self.comm.download_bytes_total),
            "upload_keys_total_run": int(self.comm.upload_keys_total),
            "download_keys_total_run": int(self.comm.download_keys_total),
            "comm_bytes_total_run": int(self.comm.upload_bytes_total + self.comm.download_bytes_total),
            "wall_time_sec": wall_time_sec,
            "final_avg_accuracy": float(final_cl_metrics["final_avg_accuracy"]),
            "final_avg_forgetting": float(final_cl_metrics["avg_forgetting"]),
            "forgetting_per_task": final_cl_metrics["forgetting_per_task"],
            "final_avg_bwt": float(final_cl_metrics["avg_bwt"]),
            "bwt_per_task": final_cl_metrics["bwt_per_task"],
            "final_avg_fwt": float(final_fwt_metrics["avg_fwt"]),
            "fwt_per_task": final_fwt_metrics["fwt_per_task"],
            "final_avg_accuracy_drop_from_peak": float(
                final_cl_metrics["avg_accuracy_drop_from_peak"]
            ),
            "accuracy_drop_from_peak_per_task": final_cl_metrics["accuracy_drop_from_peak_per_task"],
            "final_client_seen_accuracy_mean": float(last_fairness_summary["client_seen_accuracy_mean"]),
            "final_client_seen_accuracy_std": float(last_fairness_summary["client_seen_accuracy_std"]),
            "final_client_seen_accuracy_min": float(last_fairness_summary["client_seen_accuracy_min"]),
            "final_client_seen_accuracy_max": float(last_fairness_summary["client_seen_accuracy_max"]),
            "final_client_seen_accuracy_per_client": last_fairness_summary["client_seen_accuracy_per_client"],
            "task_accuracy_matrix_path": self.task_matrix_path,
            "final_model_path": self.final_model_path,
            "metrics_path": self.metrics_path,
            "manifest_path": self.manifest_path,
            "tegrastats_log_path": power_summary.logfile_path,
            "tegrastats_found": bool(power_summary.tegrastats_found),
            "tegrastats_sample_count": int(power_summary.sample_count),
            "power_source": str(power_summary.power_source),
            "avg_power_w": _safe_float(power_summary.avg_power_w),
            "peak_power_w": _safe_float(power_summary.peak_power_w),
            "energy_j": _safe_float(power_summary.energy_j),
            "avg_cpu_util": _safe_float(power_summary.avg_cpu_util),
            "peak_cpu_util": _safe_float(power_summary.peak_cpu_util),
            "avg_gpu_util": _safe_float(power_summary.avg_gpu_util),
            "peak_gpu_util": _safe_float(power_summary.peak_gpu_util),
            "avg_ram_mb": _safe_float(power_summary.avg_ram_mb),
            "peak_ram_mb": _safe_float(power_summary.peak_ram_mb),
            "avg_swap_mb": _safe_float(power_summary.avg_swap_mb),
            "peak_swap_mb": _safe_float(power_summary.peak_swap_mb),
        })

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return summary