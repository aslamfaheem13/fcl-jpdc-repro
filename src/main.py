#main.py
import os
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# Set BEFORE importing numpy/torch for stability/performance on edge systems
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
import random
from datetime import datetime

import numpy as np
import torch

try:
    from .config import ExperimentConfig  # type: ignore
except Exception:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from src.config import ExperimentConfig  # type: ignore


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Federated Continual Learning Experiments")

    # ---------------- Dataset / method control ----------------
    p.add_argument(
        "--dataset",
        default="cifar100",
        choices=["cifar10", "cifar100", "digit5", "tinyimagenet"],
    )
    p.add_argument(
        "--method",
        default="ALL",
        choices=[
            "ALL",
            "FULL_FEDAVG",
            "TRUE_FEDAVG",
            "SHARED_ADAPTER",
            "TASK_ADAPTER",
            "LWF",
            "EWC",
            "MAS",
            "SI",
        ],
        help="Run ALL methods or a single method",
    )

    # ---------------- FL algorithm control ----------------
    p.add_argument(
        "--fl_algo",
        default="FEDAVG",
        choices=["FEDAVG", "FEDPROX", "SCAFFOLD", "LOCAL_ONLY", "JOINT_ONLY"],
        help=(
            "FL optimization algorithm. FEDPROX uses --mu. "
            "SCAFFOLD uses control variates. LOCAL_ONLY disables federation. "
            "JOINT_ONLY is the upper-bound centralized baseline."
        ),
    )

    # ---------------- Experiment knobs ----------------
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44], help="List of seeds")
    p.add_argument("--alpha", type=float, default=0.7, help="Dirichlet alpha for non-IID")
    p.add_argument("--iid", type=int, default=0, help="1 = IID, 0 = non-IID")
    p.add_argument("--rounds_per_task", type=int, default=10, help="Federated rounds per task")
    p.add_argument("--local_epochs", type=int, default=3, help="Local epochs per client per round")
    p.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    p.add_argument("--batch_size", type=int, default=64, help="Train batch size")
    p.add_argument(
        "--adapter_bottleneck",
        type=int,
        default=16,
        help="Adapter bottleneck dimension for shared/task adapters.",
    )

    # quick smoke mode
    p.add_argument("--smoke", type=int, default=0, help="1 = tiny run (2 tasks, 1 round/task, 1 epoch, 2 clients)")

    # Data split safety knobs
    p.add_argument("--min_samples", type=int, default=2, help="Min samples per client per task")
    p.add_argument("--max_tries", type=int, default=30, help="Max tries for Dirichlet split")

    # Data root
    p.add_argument("--data_root", type=str, default=None, help="Dataset root path (recommended)")
    p.add_argument("--data-root", dest="data_root", type=str, help=argparse.SUPPRESS)

    # Output directory
    p.add_argument("--out_dir", type=str, default="experiments/latest", help="Output directory root for this run")
    p.add_argument("--out-dir", dest="out_dir", type=str, help=argparse.SUPPRESS)

    # Optimizer knobs
    p.add_argument("--lr", type=float, default=5e-4, help="Client learning rate")
    p.add_argument("--weight_decay", type=float, default=0.0, help="Client weight decay")

    # FedProx
    p.add_argument("--mu", type=float, default=0.0, help="FedProx strength (0 disables)")

    # Client env knobs
    p.add_argument(
        "--head_train_mode",
        type=str,
        default="current",
        choices=["current", "seen"],
        help="Head training mode: 'current' trains only current head, 'seen' trains heads up to current task.",
    )
    p.add_argument("--grad_clip", type=float, default=0.0, help="If >0, sets FCL_GRAD_CLIP env.")
    p.add_argument("--debug", type=int, default=0, help="If 1, sets FCL_DEBUG=1 to print trainable params.")

    # Replay baseline
    p.add_argument("--replay", type=int, default=0, help="1 enables replay buffer baseline, 0 disables")
    p.add_argument("--replay_per_class", type=int, default=20, help="Replay buffer slots per label per client")
    p.add_argument("--replay_batch_size", type=int, default=32, help="Replay samples per optimizer step")
    p.add_argument("--replay_lambda", type=float, default=1.0, help="Weight for replay loss term")

    # Power controls
    p.add_argument("--enable_power", type=int, default=1, help="1 enables power instrumentation flag, 0 disables")
    p.add_argument("--tegrastats_interval_ms", type=int, default=500, help="tegrastats sampling interval in ms")
    p.add_argument(
        "--tegrastats_bin",
        type=str,
        default="tegrastats",
        help="Path to tegrastats binary (default assumes it is in PATH on Jetson).",
    )

    # Device
    p.add_argument("--device", type=str, default=None, help="e.g. 'cuda', 'cpu' (default auto)")

    # Warm-start backbone
    p.add_argument("--backbone_ckpt_in", type=str, default=None, help="Load backbone from this ckpt before training")
    p.add_argument("--backbone_ckpt_out", type=str, default=None, help="Save backbone to this ckpt after training")
    p.add_argument(
        "--pretrained_backbone",
        type=int,
        default=0,
        help="1 loads ImageNet-pretrained ResNet-18 weights before adapting the CIFAR-style backbone stem.",
    )

    # Run stamp
    p.add_argument("--run_stamp", type=str, default=None, help="Optional run stamp string (e.g., 20260304_120000).")

    # Allow download toggle
    p.add_argument("--allow_data_download", type=int, default=0, help="If 1, sets FCL_ALLOW_DATA_DOWNLOAD=1.")

    return p.parse_args()


def method_display_name(method: str) -> str:
    m = method.upper()
    if m == "FULL_FEDAVG":
        return "FROZEN_BACKBONE_HEAD_ONLY (was FULL_FEDAVG)"
    if m == "TRUE_FEDAVG":
        return "FEDAVG_FULL_MODEL (TRUE_FEDAVG)"
    if m == "EWC":
        return "ELASTIC_WEIGHT_CONSOLIDATION"
    if m == "MAS":
        return "MEMORY_AWARE_SYNAPSES"
    if m == "SI":
        return "SYNAPTIC_INTELLIGENCE"
    return m


def method_trainable_scope(method: str) -> str:
    m = str(method).upper()
    if m == "FULL_FEDAVG":
        return "head_only"
    if m in ("TRUE_FEDAVG", "LWF", "EWC", "MAS", "SI"):
        return "full_model_no_adapters"
    if m == "SHARED_ADAPTER":
        return "shared_adapter_plus_heads"
    if m == "TASK_ADAPTER":
        return "task_adapter_plus_heads"
    return "unknown"


def method_backbone_frozen(method: str) -> bool:
    return str(method).upper() == "FULL_FEDAVG"


def evaluation_protocol(dataset_name: str) -> str:
    dn = str(dataset_name).lower()
    if dn in ("cifar10", "cifar100", "tinyimagenet"):
        return "task_incremental_class_sequence"
    if dn == "digit5":
        return "task_incremental_domain_sequence"
    return "task_incremental"


def task_id_known_at_inference(dataset_name: str) -> bool:
    return True


def _apply_client_env_from_args(args):
    os.environ["FCL_HEAD_TRAIN_MODE"] = str(args.head_train_mode).lower()
    os.environ["FCL_GRAD_CLIP"] = str(float(args.grad_clip))
    os.environ["FCL_DEBUG"] = "1" if int(args.debug) == 1 else "0"
    os.environ["FCL_ALLOW_DATA_DOWNLOAD"] = "1" if int(args.allow_data_download) == 1 else "0"


def _import_runtime():
    try:
        from .model import SimpleCNN  # type: ignore
        from .data import get_clients_data, get_dataset_num_classes, get_default_task_layout  # type: ignore
        from .client import Client  # type: ignore
        from .server import Server  # type: ignore
        return SimpleCNN, get_clients_data, get_dataset_num_classes, get_default_task_layout, Client, Server
    except Exception:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from src.model import SimpleCNN  # type: ignore
        from src.data import get_clients_data, get_dataset_num_classes, get_default_task_layout  # type: ignore
        from src.client import Client  # type: ignore
        from src.server import Server  # type: ignore
        return SimpleCNN, get_clients_data, get_dataset_num_classes, get_default_task_layout, Client, Server


def _auto_find_data_root(dataset_name: str) -> Optional[str]:
    candidates = [
        os.environ.get("FCL_DATA_ROOT", "").strip() or None,
        "/workspace/data",
        "/workspace/datasets",
        "/workspace/fcl_lab/data",
        os.path.join(os.getcwd(), "data"),
    ]

    expected_entries = {
        "cifar10": "cifar-10-batches-py",
        "cifar100": "cifar-100-python",
        "digit5": "digit5",
        "tinyimagenet": "tiny-imagenet-200",
    }

    expected = expected_entries[dataset_name]

    for root in candidates:
        if not root:
            continue
        if os.path.isdir(os.path.join(root, expected)):
            return root
    return None


def _cfg_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    base = asdict(cfg)
    for key in [
        "pretrained_backbone",
        "task_id_known_at_inference",
        "evaluation_protocol",
        "trainable_scope",
        "backbone_frozen",
        "tegrastats_bin",
    ]:
        if hasattr(cfg, key):
            base[key] = getattr(cfg, key)
    return base


def _validate_config(cfg: ExperimentConfig) -> None:
    valid_methods = {"FULL_FEDAVG", "TRUE_FEDAVG", "SHARED_ADAPTER", "TASK_ADAPTER", "LWF", "EWC", "MAS", "SI"}
    valid_algos = {"FEDAVG", "FEDPROX", "SCAFFOLD", "LOCAL_ONLY", "JOINT_ONLY"}
    valid_datasets = {"cifar10", "cifar100", "digit5", "tinyimagenet"}

    if cfg.method not in valid_methods:
        raise ValueError(f"Invalid method: {cfg.method}")
    if cfg.fl_algo not in valid_algos:
        raise ValueError(f"Invalid fl_algo: {cfg.fl_algo}")
    if cfg.dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset_name: {cfg.dataset_name}")

    if cfg.num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if cfg.rounds_per_task <= 0:
        raise ValueError("rounds_per_task must be > 0")
    if cfg.local_epochs <= 0:
        raise ValueError("local_epochs must be > 0")
    if cfg.num_tasks <= 0:
        raise ValueError("num_tasks must be > 0")
    if cfg.classes_per_task <= 0:
        raise ValueError("classes_per_task must be > 0")
    if cfg.num_classes <= 0:
        raise ValueError("num_classes must be > 0")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if cfg.min_samples_per_client_per_task <= 0:
        raise ValueError("min_samples_per_client_per_task must be > 0")
    if cfg.max_tries <= 0:
        raise ValueError("max_tries must be > 0")
    if int(getattr(cfg, "adapter_bottleneck", 16)) <= 0:
        raise ValueError("adapter_bottleneck must be > 0")

    if cfg.fl_algo == "FEDAVG" and cfg.mu != 0.0:
        print("[WARN] FEDAVG selected; overriding mu to 0.0 behavior at runtime.")
    if cfg.fl_algo == "FEDPROX" and cfg.mu < 0.0:
        raise ValueError("FedProx mu must be >= 0")
    if cfg.fl_algo in {"SCAFFOLD", "LOCAL_ONLY", "JOINT_ONLY"} and cfg.mu != 0.0:
        print(f"[WARN] {cfg.fl_algo} selected; overriding mu to 0.0 behavior at runtime.")

    if cfg.replay_per_class < 0:
        raise ValueError("replay_per_class must be >= 0")
    if cfg.replay_batch_size <= 0:
        raise ValueError("replay_batch_size must be > 0")
    if cfg.replay_lambda < 0.0:
        raise ValueError("replay_lambda must be >= 0")

    if cfg.dataset_name == "digit5":
        if cfg.num_classes != 10:
            raise ValueError(f"digit5 expects num_classes=10, got {cfg.num_classes}")
        if cfg.classes_per_task != 10:
            raise ValueError(f"digit5 expects classes_per_task=10, got {cfg.classes_per_task}")
        if cfg.num_tasks < 1 or cfg.num_tasks > 5:
            raise ValueError(f"digit5 expects 1 <= num_tasks <= 5, got {cfg.num_tasks}")

    if cfg.dataset_name == "tinyimagenet":
        if cfg.num_classes != 200:
            raise ValueError(f"tinyimagenet expects num_classes=200, got {cfg.num_classes}")
        if cfg.classes_per_task <= 0:
            raise ValueError("tinyimagenet expects classes_per_task > 0")
        if cfg.num_tasks * cfg.classes_per_task > cfg.num_classes:
            raise ValueError(
                f"tinyimagenet invalid layout: num_tasks({cfg.num_tasks}) * "
                f"classes_per_task({cfg.classes_per_task}) > num_classes({cfg.num_classes})"
            )


def _resolve_dataset_layout(
    get_dataset_num_classes,
    get_default_task_layout,
    dataset_name: str,
    smoke_tasks_override: Optional[int] = None,
):
    num_classes = get_dataset_num_classes(dataset_name)
    num_tasks, classes_per_task = get_default_task_layout(dataset_name)

    if smoke_tasks_override is not None:
        if dataset_name == "digit5":
            num_tasks = min(max(1, smoke_tasks_override), 5)
            classes_per_task = 10
        else:
            num_tasks = smoke_tasks_override

    if dataset_name != "digit5":
        if num_tasks * classes_per_task > num_classes:
            raise ValueError(
                f"Invalid dataset layout for {dataset_name}: "
                f"num_tasks({num_tasks}) * classes_per_task({classes_per_task}) > num_classes({num_classes})"
            )

    return num_classes, num_tasks, classes_per_task


def _resolve_methods(method_arg: str) -> List[str]:
    if method_arg == "ALL":
        return ["FULL_FEDAVG", "TRUE_FEDAVG", "SHARED_ADAPTER", "TASK_ADAPTER"]
    return [method_arg]


def _build_global_model(SimpleCNN, cfg: ExperimentConfig):
    model = SimpleCNN(
        num_tasks=cfg.num_tasks,
        num_classes=cfg.num_classes,
        bottleneck=int(getattr(cfg, "adapter_bottleneck", 16)),
        pretrained=bool(getattr(cfg, "pretrained_backbone", False)),
    )

    if cfg.backbone_ckpt_in:
        print(f"[WarmStart] Loading backbone from: {cfg.backbone_ckpt_in}")
        model.load_backbone(cfg.backbone_ckpt_in, strict_backbone=True)

    if cfg.method == "FULL_FEDAVG":
        model.freeze_backbone()
    else:
        model.unfreeze_backbone()

    device = cfg.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(torch.device(device))
    return model


def _split_clients_data_output(clients_data_output, num_clients: int):
    if isinstance(clients_data_output, tuple) and len(clients_data_output) == 2:
        train_clients_data, test_clients_data = clients_data_output
    else:
        train_clients_data = clients_data_output
        test_clients_data = None

    if len(train_clients_data) != num_clients:
        raise RuntimeError(f"Expected {num_clients} client train-loader groups, got {len(train_clients_data)}")

    if test_clients_data is not None and len(test_clients_data) != num_clients:
        raise RuntimeError(f"Expected {num_clients} client test-loader groups, got {len(test_clients_data)}")

    return train_clients_data, test_clients_data


def _client_fl_algo(cfg: ExperimentConfig) -> str:
    if str(cfg.fl_algo).upper() in {"LOCAL_ONLY", "JOINT_ONLY"}:
        return "FEDAVG"
    return str(cfg.fl_algo).upper()


def _build_clients(Client, global_model, train_clients_data, test_clients_data, cfg: ExperimentConfig):
    clients = []

    client_algo = _client_fl_algo(cfg)

    for client_id, train_loaders in enumerate(train_clients_data):
        print(f"[DEBUG] Passing classes_per_task={cfg.classes_per_task} to Client {client_id}")
        client_model = copy.deepcopy(global_model)
        test_loaders = None
        if test_clients_data is not None:
            test_loaders = test_clients_data[client_id]
            

        client = Client(
            client_id=client_id,
            model=client_model,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            classes_per_task=cfg.classes_per_task,
            method=cfg.method,
            fl_algo=client_algo,
            mu=(cfg.mu if client_algo == "FEDPROX" else 0.0),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            device=cfg.device,
            replay=cfg.replay,
            replay_per_class=cfg.replay_per_class,
            replay_batch_size=cfg.replay_batch_size,
            replay_lambda=cfg.replay_lambda,
            
        )
        clients.append(client)

    return clients


def _print_launch_summary(cfg: ExperimentConfig):
    print("\n================= EXPERIMENT CONFIG =================")
    print("Method:", cfg.method, f"({method_display_name(cfg.method)})")
    print("Dataset:", cfg.dataset_name)
    print("Seed:", cfg.seed)
    print("IID:", cfg.iid)
    print("Alpha:", cfg.alpha)
    print("Num clients:", cfg.num_clients)
    print("Num tasks:", cfg.num_tasks)
    print("Classes/task:", cfg.classes_per_task)
    print("Rounds/task:", cfg.rounds_per_task)
    print("Local epochs:", cfg.local_epochs)
    print("Batch size:", cfg.batch_size)
    print("Adapter bottleneck:", getattr(cfg, "adapter_bottleneck", 16))
    print("FL Algo:", cfg.fl_algo)
    print("Mu:", cfg.mu)
    print("LR:", cfg.lr)
    print("Weight decay:", cfg.weight_decay)
    print("Replay enabled:", cfg.replay)
    print("Replay per class:", cfg.replay_per_class)
    print("Replay batch size:", cfg.replay_batch_size)
    print("Replay lambda:", cfg.replay_lambda)
    print("Power enabled:", cfg.enable_power)
    print("Tegrastats interval (ms):", cfg.tegrastats_interval_ms)
    print("Tegrastats bin:", getattr(cfg, "tegrastats_bin", "tegrastats"))
    print("Data root:", cfg.data_root)
    print("Out dir:", cfg.out_dir)
    print("Run stamp:", cfg.run_stamp)
    print("Device override:", cfg.device)
    print("WarmStart ckpt_in:", cfg.backbone_ckpt_in)
    print("WarmStart ckpt_out:", cfg.backbone_ckpt_out)
    print("Pretrained backbone:", bool(getattr(cfg, "pretrained_backbone", False)))
    print("Trainable scope:", getattr(cfg, "trainable_scope", "unknown"))
    print("Backbone frozen:", bool(getattr(cfg, "backbone_frozen", False)))
    print("Evaluation protocol:", getattr(cfg, "evaluation_protocol", "unknown"))
    print("Task ID known at inference:", bool(getattr(cfg, "task_id_known_at_inference", True)))
    print("Env FCL_HEAD_TRAIN_MODE:", os.environ.get("FCL_HEAD_TRAIN_MODE"))
    print("Env FCL_GRAD_CLIP:", os.environ.get("FCL_GRAD_CLIP"))
    print("Env FCL_DEBUG:", os.environ.get("FCL_DEBUG"))
    print("Env FCL_ALLOW_DATA_DOWNLOAD:", os.environ.get("FCL_ALLOW_DATA_DOWNLOAD", "0"))
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=====================================================\n")


def _algo_tag(fl_algo: str, mu: float) -> str:
    if fl_algo == "FEDPROX":
        return f"FEDPROX_mu{mu}"
    return fl_algo


def _replay_tag(cfg: ExperimentConfig) -> str:
    if not cfg.replay:
        return "noreplay"
    return f"replay_rpc{cfg.replay_per_class}_rbs{cfg.replay_batch_size}_rl{cfg.replay_lambda}"


def _attach_extra_cfg_fields(cfg: ExperimentConfig, args) -> None:
    setattr(cfg, "pretrained_backbone", bool(args.pretrained_backbone))
    setattr(cfg, "task_id_known_at_inference", task_id_known_at_inference(cfg.dataset_name))
    setattr(cfg, "evaluation_protocol", evaluation_protocol(cfg.dataset_name))
    setattr(cfg, "trainable_scope", method_trainable_scope(cfg.method))
    setattr(cfg, "backbone_frozen", method_backbone_frozen(cfg.method))
    setattr(cfg, "tegrastats_bin", str(args.tegrastats_bin))


def main():
    args = parse_args()
    _apply_client_env_from_args(args)

    if str(args.fl_algo).upper() not in {"FEDAVG", "FEDPROX", "SCAFFOLD", "LOCAL_ONLY", "JOINT_ONLY"}:
        raise SystemExit(f"Unknown --fl_algo: {args.fl_algo}")

    (
        SimpleCNN,
        get_clients_data,
        get_dataset_num_classes,
        get_default_task_layout,
        Client,
        Server,
    ) = _import_runtime()

    dataset_name = args.dataset.lower()
    iid = bool(args.iid)
    alpha = float(args.alpha)
    fl_algo = str(args.fl_algo).upper()

    if fl_algo == "FEDAVG":
        mu = 0.0
    elif fl_algo == "FEDPROX":
        mu = float(args.mu)
    elif fl_algo in {"SCAFFOLD", "LOCAL_ONLY", "JOINT_ONLY"}:
        mu = 0.0
    else:
        raise SystemExit(f"Unknown --fl_algo: {fl_algo}")

    if int(args.smoke) == 1:
        print("[SMOKE MODE] Overriding: num_clients=2, rounds_per_task=1, local_epochs=1, tasks=2")
        num_clients = 2
        rounds_per_task = 1
        local_epochs = 1
        smoke_tasks_override = 2
    else:
        num_clients = int(args.num_clients)
        rounds_per_task = int(args.rounds_per_task)
        local_epochs = int(args.local_epochs)
        smoke_tasks_override = None

    data_root = args.data_root
    if data_root is None:
        env_root = os.environ.get("FCL_DATA_ROOT", "").strip()
        data_root = env_root if env_root else None
    if data_root is None:
        data_root = _auto_find_data_root(dataset_name)

    num_classes, num_tasks, classes_per_task = _resolve_dataset_layout(
        get_dataset_num_classes=get_dataset_num_classes,
        get_default_task_layout=get_default_task_layout,
        dataset_name=dataset_name,
        smoke_tasks_override=smoke_tasks_override,
    )

    methods = _resolve_methods(args.method)
    run_stamp = str(args.run_stamp).strip() if args.run_stamp else datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = str(args.out_dir)

    print(f"[Launch] dataset={dataset_name} methods={methods} seeds={args.seeds}")

    for seed in args.seeds:
        set_seed(int(seed))

        for method in methods:
            tmp_cfg = ExperimentConfig(
                method=method,
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_tasks=num_tasks,
                classes_per_task=classes_per_task,
                num_clients=num_clients,
                rounds_per_task=rounds_per_task,
                local_epochs=local_epochs,
                iid=iid,
                alpha=alpha,
                data_root=data_root,
                seed=int(seed),
                min_samples_per_client_per_task=int(args.min_samples),
                max_tries=int(args.max_tries),
                batch_size=int(args.batch_size),
                adapter_bottleneck=int(args.adapter_bottleneck),
                fl_algo=fl_algo,
                mu=float(mu),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                out_dir="",
                run_stamp=run_stamp,
                device=args.device,
                enable_power=bool(args.enable_power),
                tegrastats_interval_ms=int(args.tegrastats_interval_ms),
                replay=bool(args.replay),
                replay_per_class=int(args.replay_per_class),
                replay_batch_size=int(args.replay_batch_size),
                replay_lambda=float(args.replay_lambda),
                backbone_ckpt_in=args.backbone_ckpt_in,
                backbone_ckpt_out=args.backbone_ckpt_out,
                head_train_mode=str(args.head_train_mode),
                grad_clip=float(args.grad_clip),
                debug=bool(args.debug),
                allow_data_download=bool(args.allow_data_download),
            )
            _attach_extra_cfg_fields(tmp_cfg, args)

            out_dir = os.path.join(
                out_root,
                f"run_{run_stamp}",
                dataset_name,
                f"iid{int(iid)}_a{alpha}",
                _algo_tag(tmp_cfg.fl_algo, tmp_cfg.mu),
                f"pretrained{int(bool(getattr(tmp_cfg, 'pretrained_backbone', False)))}",
                _replay_tag(tmp_cfg),
                f"seed{seed}",
                method,
            )

            cfg = ExperimentConfig(
                method=method,
                dataset_name=dataset_name,
                num_classes=num_classes,
                num_tasks=num_tasks,
                classes_per_task=classes_per_task,
                num_clients=num_clients,
                rounds_per_task=rounds_per_task,
                local_epochs=local_epochs,
                iid=iid,
                alpha=alpha,
                data_root=data_root,
                seed=int(seed),
                min_samples_per_client_per_task=int(args.min_samples),
                max_tries=int(args.max_tries),
                batch_size=int(args.batch_size),
                adapter_bottleneck=int(args.adapter_bottleneck),
                fl_algo=fl_algo,
                mu=float(mu),
                lr=float(args.lr),
                weight_decay=float(args.weight_decay),
                out_dir=out_dir,
                run_stamp=run_stamp,
                device=args.device,
                enable_power=bool(args.enable_power),
                tegrastats_interval_ms=int(args.tegrastats_interval_ms),
                replay=bool(args.replay),
                replay_per_class=int(args.replay_per_class),
                replay_batch_size=int(args.replay_batch_size),
                replay_lambda=float(args.replay_lambda),
                backbone_ckpt_in=args.backbone_ckpt_in,
                backbone_ckpt_out=args.backbone_ckpt_out,
                head_train_mode=str(args.head_train_mode),
                grad_clip=float(args.grad_clip),
                debug=bool(args.debug),
                allow_data_download=bool(args.allow_data_download),
            )
            _attach_extra_cfg_fields(cfg, args)

            _validate_config(cfg)
            _print_launch_summary(cfg)
            print(f"[DEBUG] classes_per_task = {cfg.classes_per_task}")
            
            os.makedirs(cfg.out_dir, exist_ok=True)
            with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(_cfg_dict(cfg), f, indent=2)

            global_model = _build_global_model(SimpleCNN, cfg)

            clients_data_output = get_clients_data(
                num_clients=cfg.num_clients,
                num_tasks=cfg.num_tasks,
                classes_per_task=cfg.classes_per_task,
                alpha=cfg.alpha,
                iid=cfg.iid,
                dataset_name=cfg.dataset_name,
                data_root=cfg.data_root,
                batch_size=cfg.batch_size,
                min_samples_per_client_per_task=cfg.min_samples_per_client_per_task,
                max_tries=cfg.max_tries,
            )

            train_clients_data, test_clients_data = _split_clients_data_output(
                clients_data_output=clients_data_output,
                num_clients=cfg.num_clients,
            )

            if test_clients_data is None:
                print(
                    "[WARN] get_clients_data(...) returned train loaders only. "
                    "Client fairness will use local train loaders as fallback. "
                    "To enable true local-test fairness, update src/data.py so it returns "
                    "(train_clients_data, test_clients_data)."
                )
            else:
                print("[INFO] Using per-client local test loaders for fairness evaluation.")

            clients = _build_clients(
                Client=Client,
                global_model=global_model,
                train_clients_data=train_clients_data,
                test_clients_data=test_clients_data,
                cfg=cfg,
            )

            server = Server(
                model=global_model,
                clients=clients,
                cfg=cfg,
            )

            summary = server.run()

            print("\n" + "=" * 90)
            print(f"[DONE] dataset={cfg.dataset_name} method={cfg.method} seed={cfg.seed}")
            print(json.dumps(summary, indent=2))
            print("=" * 90 + "\n")

    print("\n✅ ALL EXPERIMENTS FINISHED")


if __name__ == "__main__":
    main()