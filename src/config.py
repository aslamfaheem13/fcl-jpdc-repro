#config.py
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    method: str
    dataset_name: str
    num_classes: int
    num_tasks: int
    classes_per_task: int

    num_clients: int
    rounds_per_task: int
    local_epochs: int

    iid: bool
    alpha: float
    data_root: Optional[str]
    seed: int

    min_samples_per_client_per_task: int
    max_tries: int
    batch_size: int
    adapter_bottleneck: int

    fl_algo: str
    mu: float
    lr: float
    weight_decay: float

    out_dir: str
    run_stamp: str
    device: Optional[str]

    enable_power: bool
    tegrastats_interval_ms: int

    replay: bool
    replay_per_class: int
    replay_batch_size: int
    replay_lambda: float

    backbone_ckpt_in: Optional[str]
    backbone_ckpt_out: Optional[str]

    head_train_mode: str
    grad_clip: float
    debug: bool
    allow_data_download: bool