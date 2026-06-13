#client.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import copy
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ============================================================
# Environment toggles
# ============================================================
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "")
    if v == "":
        return default
    try:
        return int(v)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "")
    if v == "":
        return default
    return str(v).strip()


FCL_DEBUG = _env_flag("FCL_DEBUG", default=False)
FCL_GRAD_CLIP = _env_float("FCL_GRAD_CLIP", default=0.0)
FCL_HEAD_TRAIN_MODE = _env_str("FCL_HEAD_TRAIN_MODE", default="current").lower()
FCL_LWF_LAMBDA = _env_float("FCL_LWF_LAMBDA", default=1.0)
FCL_LWF_TEMPERATURE = _env_float("FCL_LWF_TEMPERATURE", default=2.0)
FCL_EWC_LAMBDA = _env_float("FCL_EWC_LAMBDA", default=10.0)
FCL_EWC_FISHER_MAX_BATCHES = _env_int("FCL_EWC_FISHER_MAX_BATCHES", default=20)
FCL_MAS_LAMBDA = _env_float("FCL_MAS_LAMBDA", default=10.0)
FCL_MAS_MAX_BATCHES = _env_int("FCL_MAS_MAX_BATCHES", default=20)
FCL_SI_LAMBDA = _env_float("FCL_SI_LAMBDA", default=10.0)
FCL_SI_EPS = _env_float("FCL_SI_EPS", default=0.1)


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
            raise KeyError(f"[Client] Attempted to load unknown key: {k}")
        if tuple(sd[k].shape) != tuple(v.shape):
            raise ValueError(
                f"[Client] Shape mismatch for key='{k}': model={tuple(sd[k].shape)} incoming={tuple(v.shape)}"
            )
        sd[k] = v.detach().clone()
    model.load_state_dict(sd, strict=False)


# ============================================================
# Update object
# ============================================================
@dataclass
class ClientUpdate:
    client_id: int
    num_samples: int
    state_dict: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    communication_bytes: int
    client_control_delta: Optional[Dict[str, torch.Tensor]] = None


# ============================================================
# Client
# ============================================================
class Client:
    """
    Client trains on per-task DataLoaders.

    Methods:
      - FULL_FEDAVG: head-only baseline (train classifiers only), adapter_mode="none"
      - TRUE_FEDAVG: train backbone + heads, exclude adapters, adapter_mode="none"
      - SHARED_ADAPTER: train shared adapter + heads, adapter_mode="shared"
      - TASK_ADAPTER: train only current task adapter + heads, adapter_mode="task"
      - LWF: full-model training + Learning without Forgetting distillation, adapter_mode="none"
      - EWC: full-model training + Elastic Weight Consolidation regularization, adapter_mode="none"
      - MAS: full-model training + Memory Aware Synapses regularization, adapter_mode="none"
      - SI: full-model training + Synaptic Intelligence regularization, adapter_mode="none"

    FL algos:
      - FEDAVG
      - FEDPROX
      - SCAFFOLD
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loaders: List,
        test_loaders: Optional[List] = None,
        classes_per_task: Optional[int] = None,
        method: str = "FULL_FEDAVG",
        fl_algo: str = "FEDAVG",
        mu: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[str] = None,
        replay: bool = False,
        replay_per_class: int = 20,
        replay_batch_size: int = 32,
        replay_lambda: float = 1.0,
    ):
        self.client_id = int(client_id)
        self.model = model
        self.train_loaders = list(train_loaders)
        self.test_loaders = list(test_loaders) if test_loaders is not None else [None] * len(self.train_loaders)
        self.method = str(method).upper()
        self.fl_algo = str(fl_algo).upper()
        self.mu = float(mu)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.classes_per_task = classes_per_task

        if len(self.test_loaders) != len(self.train_loaders):
            raise ValueError(
                f"[Client] test_loaders length ({len(self.test_loaders)}) must match "
                f"train_loaders length ({len(self.train_loaders)})"
            )

        if device is None:
            self.device = next(self.model.parameters()).device
        else:
            self.device = torch.device(device)
            self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.replay = bool(replay)
        self.replay_per_class = int(replay_per_class)
        self.replay_batch_size = int(replay_batch_size)
        self.replay_lambda = float(replay_lambda)

        self.replay_buffer: Dict[int, List[Tuple[torch.Tensor, int, int]]] = {}

        self.optimizer: Optional[optim.Optimizer] = None
        self._opt_task_id: Optional[int] = None

        self.client_control: Dict[str, torch.Tensor] = {}
        self.server_control: Dict[str, torch.Tensor] = {}

        # LwF state
        self.teacher_model: Optional[nn.Module] = None
        self._teacher_task_id: int = -1

        # EWC state
        self.ewc_fisher: Dict[str, torch.Tensor] = {}
        self.ewc_anchor: Dict[str, torch.Tensor] = {}
        self._ewc_consolidated_upto_task: int = -1

        # MAS state
        self.mas_importance: Dict[str, torch.Tensor] = {}
        self.mas_anchor: Dict[str, torch.Tensor] = {}
        self._mas_consolidated_upto_task: int = -1

        # SI state
        self.si_omega: Dict[str, torch.Tensor] = {}
        self.si_anchor: Dict[str, torch.Tensor] = {}
        self.si_running_contrib: Dict[str, torch.Tensor] = {}
        self.si_prev_params: Dict[str, torch.Tensor] = {}
        self._si_consolidated_upto_task: int = -1

    # ---------------- Label mapping ----------------
    def _to_local_labels(self, y: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Convert global class labels to task-local labels when task heads output
        only classes_per_task logits.

        Example for Tiny-ImageNet with 20 classes/task:
            task_id=3, global labels 60..79 -> local labels 0..19

        If classes_per_task is None, labels are returned unchanged.
        This keeps backward compatibility with full-class heads.
        """
        if self.classes_per_task is None:
            return y

        local_y = y - int(task_id) * int(self.classes_per_task)

        if torch.any(local_y < 0) or torch.any(local_y >= int(self.classes_per_task)):
            raise ValueError(
                f"[Client] Local label mapping out of range for task_id={task_id}, "
                f"classes_per_task={self.classes_per_task}. "
                f"Original labels: min={int(y.min().item())}, max={int(y.max().item())}; "
                f"Local labels: min={int(local_y.min().item())}, max={int(local_y.max().item())}"
            )

        return local_y

    # ---------------- Debug ----------------
    def _debug_print_trainable(self, tag: str):
        if not FCL_DEBUG:
            return
        print(f"\n[DEBUG][Client {self.client_id}][{self.method}] Trainable parameters ({tag}):")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print("   ", name)
        print("")

    # ---------------- Method routing ----------------
    def _adapter_mode(self) -> str:
        if self.method in ("FULL_FEDAVG", "TRUE_FEDAVG", "LWF", "EWC", "MAS", "SI"):
            return "none"
        if self.method == "SHARED_ADAPTER":
            return "shared"
        if self.method == "TASK_ADAPTER":
            return "task"
        raise ValueError(f"[Client] Unknown method: {self.method}")

    # ---------------- Freezing / trainable selection ----------------
    def _freeze_all(self):
        for _, p in self.model.named_parameters():
            p.requires_grad = False

    def _enable_params_by_rule(self, rule_fn):
        params = []
        for name, p in self.model.named_parameters():
            if rule_fn(name):
                p.requires_grad = True
                params.append(p)
        return params

    def _head_rule(self, task_id: int):
        mode = FCL_HEAD_TRAIN_MODE
        if mode not in ("current", "seen"):
            mode = "current"

        if mode == "current":
            prefix = f"classifiers.{int(task_id)}."
            return lambda n: n.startswith(prefix)

        def rule(n: str) -> bool:
            if not n.startswith("classifiers."):
                return False
            parts = n.split(".")
            if len(parts) < 3:
                return False
            try:
                t = int(parts[1])
            except Exception:
                return False
            return t <= int(task_id)

        return rule

    def _select_trainable_params(self, task_id: int):
        if task_id is None:
            raise ValueError("[Client] task_id is required")

        self._freeze_all()
        head_rule = self._head_rule(task_id)

        if self.method == "FULL_FEDAVG":
            return self._enable_params_by_rule(lambda n: head_rule(n))

        if self.method in ("TRUE_FEDAVG", "LWF", "EWC", "MAS", "SI"):
            def rule(n: str) -> bool:
                if "shared_adapter" in n:
                    return False
                if "task_adapters" in n:
                    return False
                if n.startswith("classifiers."):
                    return head_rule(n)
                return True
            return self._enable_params_by_rule(rule)

        if self.method == "SHARED_ADAPTER":
            return self._enable_params_by_rule(lambda n: ("shared_adapter" in n) or head_rule(n))

        if self.method == "TASK_ADAPTER":
            def rule(n: str) -> bool:
                if head_rule(n):
                    return True
                return n.startswith(f"task_adapters.{int(task_id)}.")
            return self._enable_params_by_rule(rule)

        raise ValueError(f"[Client] Unknown method: {self.method}")

    def _maybe_rebuild_optimizer_for_task(self, task_id: int):
        if self._opt_task_id == task_id and self.optimizer is not None:
            return

        train_params = self._select_trainable_params(task_id=task_id)
        if len(train_params) == 0:
            raise RuntimeError(f"[Client] No trainable parameters for method={self.method} task_id={task_id}")

        if self.fl_algo == "SCAFFOLD":
            self.optimizer = optim.SGD(
                train_params,
                lr=self.lr,
                momentum=0.0,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                train_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        self._opt_task_id = task_id
        self._debug_print_trainable(tag=f"task_id={task_id}")

    # ---------------- Broadcast load / partial export ----------------
    def load_global_state(self, partial_state: Dict[str, torch.Tensor]) -> None:
        _load_partial_state(self.model, partial_state)

    def load_server_control(self, control_state: Dict[str, torch.Tensor]) -> None:
        self.server_control = {
            k: v.detach().clone().to(self.device, dtype=torch.float32)
            for k, v in control_state.items()
        }

    def init_client_control_if_needed(self, keys: List[str]) -> None:
        if self.client_control:
            return

        sd = self.model.state_dict()
        self.client_control = {}
        for k in keys:
            if k not in sd:
                raise KeyError(f"[Client] Missing key for client control init: {k}")
            self.client_control[k] = torch.zeros_like(
                sd[k],
                device=self.device,
                dtype=torch.float32,
            )

    def export_partial_state(self, keys: List[str]) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k not in sd:
                raise KeyError(f"[Client] Missing state key during export: {k}")
            out[k] = sd[k].detach().clone()
        return out

    def _snapshot_sync_params(self, sync_keys: List[str]) -> Dict[str, torch.Tensor]:
        sd = self.model.state_dict()
        snap: Dict[str, torch.Tensor] = {}
        for k in sync_keys:
            if k not in sd:
                raise KeyError(f"[Client] Missing state key during snapshot: {k}")
            snap[k] = sd[k].detach().clone().to(self.device, dtype=torch.float32)
        return snap

    # ---------------- Replay helpers ----------------
    def replay_num_samples(self) -> int:
        return sum(len(v) for v in self.replay_buffer.values())

    def replay_memory_bytes(self) -> int:
        total = 0
        for items in self.replay_buffer.values():
            for x, _, _ in items:
                total += _tensor_nbytes(x)
                total += 8
        return total

    def _buffer_add_batch(self, x: torch.Tensor, y: torch.Tensor, task_id: int):
        if not self.replay:
            return

        x_cpu = x.detach().to("cpu")
        y_cpu = y.detach().to("cpu")

        for xi, yi in zip(x_cpu, y_cpu):
            lab = int(yi.item())
            self.replay_buffer.setdefault(lab, [])

            item = (xi.clone(), lab, int(task_id))
            if len(self.replay_buffer[lab]) < self.replay_per_class:
                self.replay_buffer[lab].append(item)
            else:
                j = random.randint(0, self.replay_per_class - 1)
                self.replay_buffer[lab][j] = item

    def _buffer_sample_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not self.replay_buffer:
            return None

        labels = list(self.replay_buffer.keys())
        if len(labels) == 0:
            return None

        xs, ys, ts = [], [], []
        for _ in range(self.replay_batch_size):
            lab = random.choice(labels)
            xi, yi, ti = random.choice(self.replay_buffer[lab])
            xs.append(xi)
            ys.append(yi)
            ts.append(ti)

        x = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        t = torch.tensor(ts, dtype=torch.long)
        return x, y, t

    # ---------------- FedProx ----------------
    def _fedprox_term(self, global_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        prox = torch.tensor(0.0, device=self.device)
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                prox += torch.sum((p - global_params[name]) ** 2)
        return prox

    # ---------------- SCAFFOLD ----------------
    def _apply_scaffold_correction(self, sync_keys: List[str]) -> None:
        if self.fl_algo != "SCAFFOLD":
            return

        for name, p in self.model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if name not in sync_keys:
                continue
            if name not in self.client_control or name not in self.server_control:
                continue

            correction = self.client_control[name] - self.server_control[name]
            p.grad.data = p.grad.data + correction.to(p.grad.device, dtype=p.grad.dtype)

    # ---------------- LwF helpers ----------------
    def _refresh_teacher_for_task(self, task_id: int) -> None:
        if self.method != "LWF":
            return

        if task_id <= 0:
            self.teacher_model = None
            self._teacher_task_id = -1
            return

        if self.teacher_model is not None and self._teacher_task_id == int(task_id):
            return

        teacher = copy.deepcopy(self.model).to(self.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        self.teacher_model = teacher
        self._teacher_task_id = int(task_id)

    def _lwf_distillation_loss(self, x: torch.Tensor, current_task_id: int, adapter_mode: str) -> torch.Tensor:
        if self.method != "LWF":
            return torch.tensor(0.0, device=self.device)
        if self.teacher_model is None:
            return torch.tensor(0.0, device=self.device)
        if current_task_id <= 0:
            return torch.tensor(0.0, device=self.device)

        T = float(FCL_LWF_TEMPERATURE)
        if T <= 0.0:
            T = 2.0

        total_kd = torch.tensor(0.0, device=self.device)
        num_heads = 0

        with torch.no_grad():
            teacher_logits_by_task = []
            for prev_task_id in range(int(current_task_id)):
                t_logits = self.teacher_model(x, task_id=prev_task_id, adapter_mode=adapter_mode)
                teacher_logits_by_task.append(t_logits)

        for prev_task_id in range(int(current_task_id)):
            student_logits = self.model(x, task_id=prev_task_id, adapter_mode=adapter_mode)
            teacher_logits = teacher_logits_by_task[prev_task_id]

            kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="batchmean",
            ) * (T * T)

            total_kd = total_kd + kd
            num_heads += 1

        if num_heads <= 0:
            return torch.tensor(0.0, device=self.device)

        return total_kd / float(num_heads)

    # ---------------- EWC / MAS / SI helpers ----------------
    def _reg_param_in_scope(self, name: str, upto_task_id: int) -> bool:
        if "shared_adapter" in name:
            return False
        if "task_adapters" in name:
            return False

        if name.startswith("classifiers."):
            parts = name.split(".")
            if len(parts) < 3:
                return False
            try:
                t = int(parts[1])
            except Exception:
                return False
            return t <= int(upto_task_id)

        return True

    def _maybe_consolidate_ewc_before_task(self, task_id: int, adapter_mode: str) -> None:
        if self.method != "EWC":
            return
        if task_id <= 0:
            return

        prev_task_id = int(task_id) - 1
        if self._ewc_consolidated_upto_task >= prev_task_id:
            return

        loader = self.train_loaders[prev_task_id]
        if loader is None:
            return

        was_training = self.model.training
        self.model.eval()

        fisher: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if self._reg_param_in_scope(name, prev_task_id):
                fisher[name] = torch.zeros_like(p.detach(), device=self.device)

        n_batches = 0
        max_batches = max(1, int(FCL_EWC_FISHER_MAX_BATCHES))

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(x, task_id=prev_task_id, adapter_mode=adapter_mode)
            y_local = self._to_local_labels(y, prev_task_id)
            loss = self.criterion(logits, y_local)
            loss.backward()

            for name, p in self.model.named_parameters():
                if name in fisher and p.grad is not None:
                    fisher[name] += p.grad.detach().pow(2)

            n_batches += 1
            if n_batches >= max_batches:
                break

        if n_batches > 0:
            for name in fisher.keys():
                fisher[name] = fisher[name] / float(n_batches)

        for name, p in self.model.named_parameters():
            if name in fisher:
                self.ewc_fisher[name] = fisher[name].detach().clone()
                self.ewc_anchor[name] = p.detach().clone()

        self._ewc_consolidated_upto_task = prev_task_id

        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()

    def _ewc_penalty(self) -> torch.Tensor:
        if self.method != "EWC":
            return torch.tensor(0.0, device=self.device)
        if not self.ewc_fisher or not self.ewc_anchor:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.ewc_fisher or name not in self.ewc_anchor:
                continue
            penalty = penalty + torch.sum(self.ewc_fisher[name] * (p - self.ewc_anchor[name]).pow(2))
        return penalty

    def _maybe_consolidate_mas_before_task(self, task_id: int, adapter_mode: str) -> None:
        if self.method != "MAS":
            return
        if task_id <= 0:
            return

        prev_task_id = int(task_id) - 1
        if self._mas_consolidated_upto_task >= prev_task_id:
            return

        loader = self.train_loaders[prev_task_id]
        if loader is None:
            return

        was_training = self.model.training
        self.model.eval()

        importance: Dict[str, torch.Tensor] = {}
        for name, p in self.model.named_parameters():
            if self._reg_param_in_scope(name, prev_task_id):
                importance[name] = torch.zeros_like(p.detach(), device=self.device)

        n_batches = 0
        max_batches = max(1, int(FCL_MAS_MAX_BATCHES))

        for x, _y in loader:
            x = x.to(self.device, non_blocking=True)

            self.model.zero_grad(set_to_none=True)
            logits = self.model(x, task_id=prev_task_id, adapter_mode=adapter_mode)
            output_norm = torch.sum(logits.pow(2), dim=1).mean()
            output_norm.backward()

            for name, p in self.model.named_parameters():
                if name in importance and p.grad is not None:
                    importance[name] += p.grad.detach().abs()

            n_batches += 1
            if n_batches >= max_batches:
                break

        if n_batches > 0:
            for name in importance.keys():
                importance[name] = importance[name] / float(n_batches)

        for name, p in self.model.named_parameters():
            if name in importance:
                self.mas_importance[name] = importance[name].detach().clone()
                self.mas_anchor[name] = p.detach().clone()

        self._mas_consolidated_upto_task = prev_task_id

        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()

    def _mas_penalty(self) -> torch.Tensor:
        if self.method != "MAS":
            return torch.tensor(0.0, device=self.device)
        if not self.mas_importance or not self.mas_anchor:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.mas_importance or name not in self.mas_anchor:
                continue
            penalty = penalty + torch.sum(self.mas_importance[name] * (p - self.mas_anchor[name]).pow(2))
        return penalty

    def _si_prepare_for_task(self, task_id: int) -> None:
        if self.method != "SI":
            return

        # At start of task 0, initialize anchors and accumulators.
        if task_id == 0 and not self.si_anchor:
            for name, p in self.model.named_parameters():
                if self._reg_param_in_scope(name, task_id):
                    self.si_anchor[name] = p.detach().clone()
                    self.si_omega[name] = torch.zeros_like(p.detach(), device=self.device)
                    self.si_running_contrib[name] = torch.zeros_like(p.detach(), device=self.device)
                    self.si_prev_params[name] = p.detach().clone()

        # At start of later tasks, consolidate previous task once.
        prev_task_id = int(task_id) - 1
        if task_id > 0 and self._si_consolidated_upto_task < prev_task_id:
            self._consolidate_si_after_task(prev_task_id)

            for name, p in self.model.named_parameters():
                if self._reg_param_in_scope(name, prev_task_id):
                    self.si_running_contrib[name] = torch.zeros_like(p.detach(), device=self.device)
                    self.si_prev_params[name] = p.detach().clone()

    def _snapshot_si_pre_step(self) -> Dict[str, torch.Tensor]:
        snap: Dict[str, torch.Tensor] = {}
        if self.method != "SI":
            return snap

        for name, p in self.model.named_parameters():
            if p.requires_grad and self._reg_param_in_scope(name, self._opt_task_id if self._opt_task_id is not None else 0):
                snap[name] = p.detach().clone()
        return snap

    def _accumulate_si_after_step(self, pre_step_params: Dict[str, torch.Tensor]) -> None:
        if self.method != "SI":
            return
        if not pre_step_params:
            return

        for name, p in self.model.named_parameters():
            if name not in pre_step_params:
                continue
            if p.grad is None:
                continue

            if name not in self.si_running_contrib:
                self.si_running_contrib[name] = torch.zeros_like(p.detach(), device=self.device)

            delta = p.detach() - pre_step_params[name]
            contrib = -p.grad.detach() * delta
            self.si_running_contrib[name] = self.si_running_contrib[name] + contrib

    def _consolidate_si_after_task(self, finished_task_id: int) -> None:
        if self.method != "SI":
            return

        eps = float(FCL_SI_EPS)
        if eps <= 0.0:
            eps = 0.1

        for name, p in self.model.named_parameters():
            if not self._reg_param_in_scope(name, finished_task_id):
                continue

            current = p.detach().clone()

            if name not in self.si_anchor:
                self.si_anchor[name] = current.clone()
            if name not in self.si_omega:
                self.si_omega[name] = torch.zeros_like(current, device=self.device)
            if name not in self.si_running_contrib:
                self.si_running_contrib[name] = torch.zeros_like(current, device=self.device)

            delta_total = current - self.si_anchor[name]
            omega_add = self.si_running_contrib[name] / (delta_total.pow(2) + eps)
            omega_add = torch.clamp(omega_add, min=0.0)

            self.si_omega[name] = self.si_omega[name] + omega_add
            self.si_anchor[name] = current.clone()

        self._si_consolidated_upto_task = int(finished_task_id)

    def _si_penalty(self) -> torch.Tensor:
        if self.method != "SI":
            return torch.tensor(0.0, device=self.device)
        if not self.si_omega or not self.si_anchor:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.si_omega or name not in self.si_anchor:
                continue
            penalty = penalty + torch.sum(self.si_omega[name] * (p - self.si_anchor[name]).pow(2))
        return penalty

    # ---------------- Local fairness evaluation ----------------
    def _get_eval_loader_for_task(self, task_id: int):
        if task_id < len(self.test_loaders) and self.test_loaders[task_id] is not None:
            return self.test_loaders[task_id]
        if task_id < len(self.train_loaders):
            return self.train_loaders[task_id]
        return None

    def evaluate_seen_local(self, current_task_id: int) -> Dict[str, float]:
        self.model.eval()
        adapter_mode = self._adapter_mode()

        total_correct = 0
        total_samples = 0
        used_test_loader = False

        with torch.no_grad():
            for task_id in range(current_task_id + 1):
                loader = self._get_eval_loader_for_task(task_id)
                if loader is None:
                    continue

                if task_id < len(self.test_loaders) and self.test_loaders[task_id] is not None:
                    used_test_loader = True

                for x, y in loader:
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    logits = self.model(x, task_id=task_id, adapter_mode=adapter_mode)
                    y_local = self._to_local_labels(y, task_id)
                    pred = logits.argmax(dim=1)

                    total_correct += int((pred == y_local).sum().item())
                    total_samples += int(y.numel())

        acc = float(total_correct / total_samples) if total_samples > 0 else 0.0

        return {
            "client_id": float(self.client_id),
            "seen_local_accuracy": acc,
            "seen_local_num_samples": float(total_samples),
            "used_test_loader": 1.0 if used_test_loader else 0.0,
        }

    # ---------------- Train ----------------
    def train(self, task_id: int, epochs: int = 1, sync_keys: Optional[List[str]] = None) -> ClientUpdate:
        self.model.train()

        if task_id < 0 or task_id >= len(self.train_loaders):
            raise IndexError(f"[Client] task_id={task_id} out of range (0..{len(self.train_loaders)-1})")

        loader = self.train_loaders[task_id]
        if loader is None:
            raise ValueError(f"[Client] Train loader for task_id={task_id} is None")

        self._refresh_teacher_for_task(task_id)
        adapter_mode = self._adapter_mode()
        self._maybe_consolidate_ewc_before_task(task_id=task_id, adapter_mode=adapter_mode)
        self._maybe_consolidate_mas_before_task(task_id=task_id, adapter_mode=adapter_mode)
        self._si_prepare_for_task(task_id=task_id)

        self._maybe_rebuild_optimizer_for_task(task_id)
        if self.optimizer is None:
            raise RuntimeError("[Client] Optimizer is not initialized")

        if sync_keys is None:
            raise ValueError("[Client] sync_keys is required so the server/client contract stays explicit")

        if self.fl_algo == "SCAFFOLD":
            self.init_client_control_if_needed(sync_keys)

        start_sync_state = None
        if self.fl_algo == "SCAFFOLD":
            start_sync_state = self._snapshot_sync_params(sync_keys)

        global_params = None
        if self.fl_algo == "FEDPROX" and self.mu > 0.0:
            global_params = {
                name: p.detach().clone()
                for name, p in self.model.named_parameters()
                if p.requires_grad
            }

        running_loss = 0.0
        running_sup_loss = 0.0
        running_replay_loss = 0.0
        running_lwf_loss = 0.0
        running_ewc_loss = 0.0
        running_mas_loss = 0.0
        running_si_loss = 0.0
        num_steps = 0

        for _ in range(epochs):
            for x, y in loader:
                x_cpu_for_replay = x.detach().cpu() if self.replay else None
                y_cpu_for_replay = y.detach().cpu() if self.replay else None

                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)

                logits = self.model(x, task_id=task_id, adapter_mode=adapter_mode)
                y_local = self._to_local_labels(y, task_id)
                sup_loss = self.criterion(logits, y_local)
                loss = sup_loss

                replay_loss_value = torch.tensor(0.0, device=self.device)
                if self.replay:
                    sample = self._buffer_sample_batch()
                    if sample is not None:
                        xr, yr, tr = sample
                        xr = xr.to(self.device, non_blocking=True)
                        yr = yr.to(self.device, non_blocking=True)
                        tr = tr.to(self.device, non_blocking=True)

                        loss_r = torch.tensor(0.0, device=self.device)
                        num_replay_groups = 0

                        for t_id in torch.unique(tr).tolist():
                            if int(t_id) > int(task_id):
                                continue

                            mask = (tr == t_id)
                            if int(mask.sum().item()) == 0:
                                continue

                            logits_r = self.model(
                                xr[mask],
                                task_id=int(t_id),
                                adapter_mode=adapter_mode,
                            )
                            y_local_r = self._to_local_labels(yr[mask], int(t_id))
                            loss_r = loss_r + self.criterion(logits_r, y_local_r)
                            num_replay_groups += 1

                        if num_replay_groups > 0:
                            loss_r = loss_r / float(num_replay_groups)
                            replay_loss_value = loss_r
                            loss = loss + self.replay_lambda * loss_r

                lwf_loss_value = torch.tensor(0.0, device=self.device)
                if self.method == "LWF":
                    lwf_loss_value = self._lwf_distillation_loss(
                        x=x,
                        current_task_id=task_id,
                        adapter_mode=adapter_mode,
                    )
                    loss = loss + float(FCL_LWF_LAMBDA) * lwf_loss_value

                ewc_loss_value = torch.tensor(0.0, device=self.device)
                if self.method == "EWC":
                    ewc_loss_value = self._ewc_penalty()
                    loss = loss + (float(FCL_EWC_LAMBDA) / 2.0) * ewc_loss_value

                mas_loss_value = torch.tensor(0.0, device=self.device)
                if self.method == "MAS":
                    mas_loss_value = self._mas_penalty()
                    loss = loss + (float(FCL_MAS_LAMBDA) / 2.0) * mas_loss_value

                si_loss_value = torch.tensor(0.0, device=self.device)
                if self.method == "SI":
                    si_loss_value = self._si_penalty()
                    loss = loss + (float(FCL_SI_LAMBDA) / 2.0) * si_loss_value

                if self.fl_algo == "FEDPROX" and self.mu > 0.0 and global_params is not None:
                    loss = loss + (self.mu / 2.0) * self._fedprox_term(global_params)

                pre_step_params = self._snapshot_si_pre_step()
                loss.backward()

                if self.fl_algo == "SCAFFOLD":
                    self._apply_scaffold_correction(sync_keys)

                if FCL_GRAD_CLIP and FCL_GRAD_CLIP > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        max_norm=float(FCL_GRAD_CLIP),
                    )

                self.optimizer.step()
                self._accumulate_si_after_step(pre_step_params)

                if self.replay and x_cpu_for_replay is not None and y_cpu_for_replay is not None:
                    self._buffer_add_batch(x_cpu_for_replay, y_cpu_for_replay, task_id=task_id)

                running_loss += float(loss.detach().item())
                running_sup_loss += float(sup_loss.detach().item())
                running_replay_loss += float(replay_loss_value.detach().item())
                running_lwf_loss += float(lwf_loss_value.detach().item())
                running_ewc_loss += float(ewc_loss_value.detach().item())
                running_mas_loss += float(mas_loss_value.detach().item())
                running_si_loss += float(si_loss_value.detach().item())
                num_steps += 1

        partial = self.export_partial_state(sync_keys)
        comm_bytes = _state_nbytes(partial)
        avg_loss = float(running_loss / num_steps) if num_steps > 0 else 0.0
        avg_sup_loss = float(running_sup_loss / num_steps) if num_steps > 0 else 0.0
        avg_replay_loss = float(running_replay_loss / num_steps) if num_steps > 0 else 0.0
        avg_lwf_loss = float(running_lwf_loss / num_steps) if num_steps > 0 else 0.0
        avg_ewc_loss = float(running_ewc_loss / num_steps) if num_steps > 0 else 0.0
        avg_mas_loss = float(running_mas_loss / num_steps) if num_steps > 0 else 0.0
        avg_si_loss = float(running_si_loss / num_steps) if num_steps > 0 else 0.0
        num_samples = int(len(loader.dataset))

        client_control_delta = None
        if self.fl_algo == "SCAFFOLD":
            if start_sync_state is None:
                raise RuntimeError("[Client][SCAFFOLD] Missing start_sync_state")
            if num_steps <= 0:
                raise RuntimeError("[Client][SCAFFOLD] num_steps must be > 0")
            if self.lr <= 0.0:
                raise RuntimeError("[Client][SCAFFOLD] lr must be > 0")

            client_control_delta = {}

            end_sync_state = self._snapshot_sync_params(sync_keys)
            scale = 1.0 / (float(num_steps) * float(self.lr))

            for k in sync_keys:
                if k not in self.client_control or k not in self.server_control:
                    continue

                w_global = start_sync_state[k]
                w_local = end_sync_state[k]

                ci_old = self.client_control[k]
                c = self.server_control[k]

                ci_new = ci_old - c + scale * (w_global - w_local)
                delta_i = ci_new - ci_old

                self.client_control[k] = ci_new.detach().clone()
                client_control_delta[k] = delta_i.detach().clone().cpu()

        replay_num = float(self.replay_num_samples()) if self.replay else 0.0
        replay_mem_mb = (
            float(self.replay_memory_bytes()) / (1024.0 * 1024.0)
            if self.replay
            else 0.0
        )

        return ClientUpdate(
            client_id=self.client_id,
            num_samples=num_samples,
            state_dict=partial,
            metrics={
                "train_loss": avg_loss,
                "supervised_loss": avg_sup_loss,
                "replay_loss": avg_replay_loss,
                "lwf_loss": avg_lwf_loss,
                "ewc_loss": avg_ewc_loss,
                "mas_loss": avg_mas_loss,
                "si_loss": avg_si_loss,
                "num_steps": float(num_steps),
                "replay_num_samples": replay_num,
                "replay_memory_mb": replay_mem_mb,
            },
            communication_bytes=comm_bytes,
            client_control_delta=client_control_delta,
        )