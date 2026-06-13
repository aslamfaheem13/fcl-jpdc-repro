#data.py
import os
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ---------------- Path / env helpers ----------------
def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, "").strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "y", "on")


# If True, allow torchvision to download if missing.
FCL_ALLOW_DATA_DOWNLOAD = _env_flag("FCL_ALLOW_DATA_DOWNLOAD", default=False)


def get_data_path() -> str:
    """
    Dataset root path.
    Priority:
      1) FCL_DATA_ROOT environment variable
      2) /workspace/fcl_lab/data
      3) ~/Desktop/fcl_lab/data
    """
    env_root = os.environ.get("FCL_DATA_ROOT", "").strip()
    if env_root:
        return env_root

    jetson_path = "/workspace/fcl_lab/data"
    if os.path.isdir(jetson_path):
        return jetson_path

    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    return os.path.join(desktop_path, "fcl_lab", "data")


def _normalize_data_root(data_root: Optional[str]) -> str:
    if data_root is None or str(data_root).strip() == "":
        data_root = get_data_path()
    return os.path.abspath(os.path.expanduser(str(data_root)))


def _ensure_data_root_accessible(data_root: str) -> None:
    if os.path.exists(data_root):
        if not os.path.isdir(data_root):
            raise NotADirectoryError(
                f"[Data] data_root exists but is not a directory: '{data_root}'"
            )
        return

    try:
        os.makedirs(data_root, exist_ok=True)
    except PermissionError as e:
        raise PermissionError(
            f"[Data] Cannot create data_root='{data_root}' due to permissions.\n"
            f"Fix:\n"
            f"  1) Pass a writable path with --data_root\n"
            f"  2) OR export FCL_DATA_ROOT=/path/to/data\n"
            f"  3) Example: ~/fcl_lab/data\n"
        ) from e
    except OSError as e:
        raise OSError(
            f"[Data] Failed to create data_root='{data_root}'. Error: {repr(e)}"
        ) from e


# ---------------- Dataset metadata helpers ----------------
def get_dataset_num_classes(dataset_name: str) -> int:
    dn = dataset_name.lower()
    if dn == "cifar10":
        return 10
    if dn == "cifar100":
        return 100
    if dn == "digit5":
        return 10
    if dn == "tinyimagenet":
        return 200
    raise ValueError("dataset_name must be one of: 'cifar10', 'cifar100', 'digit5', 'tinyimagenet'")


def get_default_task_layout(dataset_name: str) -> Tuple[int, int]:
    dn = dataset_name.lower()
    if dn == "cifar10":
        return 5, 2
    if dn == "cifar100":
        return 10, 10
    if dn == "digit5":
        return 5, 10
    if dn == "tinyimagenet":
        return 10, 20
    raise ValueError("dataset_name must be one of: 'cifar10', 'cifar100', 'digit5', 'tinyimagenet'")


# ---------------- Normalization ----------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

DIGIT_RGB_MEAN = (0.5, 0.5, 0.5)
DIGIT_RGB_STD = (0.5, 0.5, 0.5)

# Standard ImageNet normalization for Tiny-ImageNet with pretrained backbones
TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------- Transforms ----------------
def _digit_train_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(DIGIT_RGB_MEAN, DIGIT_RGB_STD),
    ])


def _digit_test_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(DIGIT_RGB_MEAN, DIGIT_RGB_STD),
    ])


def _digit_rgb_train_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(DIGIT_RGB_MEAN, DIGIT_RGB_STD),
    ])


def _digit_rgb_test_transform():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(DIGIT_RGB_MEAN, DIGIT_RGB_STD),
    ])


def get_transforms(dataset_name: str):
    dn = dataset_name.lower()

    if dn == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return train_tf, test_tf

    if dn == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return train_tf, test_tf

    if dn == "digit5":
        return None, None

    if dn == "tinyimagenet":
        mean, std = TINY_IMAGENET_MEAN, TINY_IMAGENET_STD
        train_tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return train_tf, test_tf

    raise ValueError("dataset_name must be one of: 'cifar10', 'cifar100', 'digit5', 'tinyimagenet'")


# ---------------- CIFAR presence checks ----------------
def _expected_cifar_folder(data_root: str, dataset_name: str) -> str:
    dn = dataset_name.lower()
    if dn == "cifar10":
        return os.path.join(data_root, "cifar-10-batches-py")
    if dn == "cifar100":
        return os.path.join(data_root, "cifar-100-python")
    raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")


def _cifar_folder_exists(data_root: str, dataset_name: str) -> bool:
    return os.path.isdir(_expected_cifar_folder(data_root, dataset_name))


def _load_cifar(dataset_name: str, data_root: str, train: bool, transform):
    data_root = _normalize_data_root(data_root)
    _ensure_data_root_accessible(data_root)

    dn = dataset_name.lower()
    if dn == "cifar10":
        cls = datasets.CIFAR10
    elif dn == "cifar100":
        cls = datasets.CIFAR100
    else:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'")

    expected_folder = _expected_cifar_folder(data_root, dataset_name)

    if not _cifar_folder_exists(data_root, dataset_name):
        if FCL_ALLOW_DATA_DOWNLOAD:
            try:
                ds = cls(data_root, train=train, download=True, transform=transform)
                return ds, True
            except Exception as e:
                raise RuntimeError(
                    f"[Data] Download was allowed but failed for dataset='{dataset_name}' into '{data_root}'.\n"
                    f"Error: {repr(e)}"
                ) from e

        raise FileNotFoundError(
            f"[Data] {dataset_name} not found under data_root='{data_root}'.\n"
            f"Expected folder:\n"
            f"  {expected_folder}\n\n"
            f"Fix:\n"
            f"  1) Pass the correct path: --data_root /path/to/data\n"
            f"  2) OR export FCL_DATA_ROOT=/path/to/data\n"
            f"  3) (Dev only) allow download: export FCL_ALLOW_DATA_DOWNLOAD=1\n"
        )

    try:
        ds = cls(data_root, train=train, download=False, transform=transform)
        return ds, False
    except Exception as e:
        raise RuntimeError(
            f"[Data] Found {dataset_name} folder under '{data_root}' but torchvision failed to read it.\n"
            f"Error: {repr(e)}\n"
            f"Expected folder:\n"
            f"  {expected_folder}\n"
        ) from e


# ---------------- Tiny-ImageNet helpers ----------------
def _tiny_root(data_root: str) -> str:
    return os.path.join(data_root, "tiny-imagenet-200")


def _tiny_train_dir(data_root: str) -> str:
    return os.path.join(_tiny_root(data_root), "train")


def _tiny_val_dir(data_root: str) -> str:
    return os.path.join(_tiny_root(data_root), "val")


def _tiny_wnids_path(data_root: str) -> str:
    return os.path.join(_tiny_root(data_root), "wnids.txt")


def _tiny_val_annotations_path(data_root: str) -> str:
    return os.path.join(_tiny_root(data_root), "val", "val_annotations.txt")


def _tinyimagenet_exists(data_root: str) -> bool:
    return (
        os.path.isdir(_tiny_train_dir(data_root))
        and os.path.isdir(_tiny_val_dir(data_root))
        and os.path.isfile(_tiny_wnids_path(data_root))
    )


def _load_tinyimagenet_class_to_idx(data_root: str) -> Dict[str, int]:
    wnids_path = _tiny_wnids_path(data_root)
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(
            f"[Data] Tiny-ImageNet wnids.txt not found at: {wnids_path}"
        )

    with open(wnids_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]

    if len(classes) != 200:
        print(f"[WARN] Expected 200 Tiny-ImageNet classes, found {len(classes)} in wnids.txt")

    return {cls_name: idx for idx, cls_name in enumerate(classes)}


class _TargetsAdapterDataset(Dataset):
    """
    Wraps torchvision datasets so they consistently expose:
      - __getitem__
      - __len__
      - .targets (list[int])
    """
    def __init__(self, base_dataset: Dataset, targets: List[int]):
        self.base_dataset = base_dataset
        self.targets = list(int(x) for x in targets)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, _ = self.base_dataset[idx]
        y = self.targets[idx]
        return x, y


class _TinyImageNetValDataset(Dataset):
    """
    Handles the standard Tiny-ImageNet validation structure:
      tiny-imagenet-200/val/images/*.JPEG
      tiny-imagenet-200/val/val_annotations.txt
    """
    def __init__(self, data_root: str, transform=None):
        self.data_root = data_root
        self.transform = transform
        self.class_to_idx = _load_tinyimagenet_class_to_idx(data_root)

        val_dir = _tiny_val_dir(data_root)
        images_dir = os.path.join(val_dir, "images")
        annotations_path = _tiny_val_annotations_path(data_root)

        if not os.path.isdir(images_dir):
            raise FileNotFoundError(
                f"[Data] Tiny-ImageNet validation images folder not found: {images_dir}"
            )
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError(
                f"[Data] Tiny-ImageNet val_annotations.txt not found: {annotations_path}"
            )

        samples = []
        targets = []

        with open(annotations_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, cls_name = parts[0], parts[1]
                if cls_name not in self.class_to_idx:
                    continue
                img_path = os.path.join(images_dir, img_name)
                if not os.path.isfile(img_path):
                    continue
                label = self.class_to_idx[cls_name]
                samples.append((img_path, label))
                targets.append(label)

        if not samples:
            raise RuntimeError(
                f"[Data] No valid Tiny-ImageNet validation samples found in {images_dir}"
            )

        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _load_tinyimagenet(data_root: str, train: bool, transform):
    data_root = _normalize_data_root(data_root)
    _ensure_data_root_accessible(data_root)

    if not _tinyimagenet_exists(data_root):
        raise FileNotFoundError(
            f"[Data] Tiny-ImageNet not found under data_root='{data_root}'.\n"
            f"Expected structure:\n"
            f"  {os.path.join(data_root, 'tiny-imagenet-200')}\n"
            f"with train/, val/, wnids.txt, words.txt\n"
        )

    if train:
        train_dir = _tiny_train_dir(data_root)
        ds = datasets.ImageFolder(root=train_dir, transform=transform)
        wrapped = _TargetsAdapterDataset(ds, list(ds.targets))
        return wrapped, False

    ds = _TinyImageNetValDataset(data_root=data_root, transform=transform)
    return ds, False


# ---------------- Digit5 helpers ----------------
class _USPSH5Dataset(Dataset):
    def __init__(self, h5_path: str, train: bool, transform=None):
        self.h5_path = h5_path
        self.train = train
        self.transform = transform

        split = "train" if train else "test"

        with h5py.File(h5_path, "r") as f:
            if split not in f:
                raise RuntimeError(
                    f"[Data] Split '{split}' not found in {h5_path}. "
                    f"Available keys: {list(f.keys())}"
                )

            grp = f[split]

            if "data" in grp:
                x = np.array(grp["data"])
            elif "X" in grp:
                x = np.array(grp["X"])
            else:
                raise RuntimeError(
                    f"[Data] Could not find image data in split '{split}'. "
                    f"Available keys: {list(grp.keys())}"
                )

            if "target" in grp:
                y = np.array(grp["target"])
            elif "y" in grp:
                y = np.array(grp["y"])
            elif "label" in grp:
                y = np.array(grp["label"])
            else:
                raise RuntimeError(
                    f"[Data] Could not find labels in split '{split}'. "
                    f"Available keys: {list(grp.keys())}"
                )

        if x.ndim == 2:
            side = int(np.sqrt(x.shape[1]))
            if side * side != x.shape[1]:
                raise RuntimeError(
                    f"[Data] USPS flat vectors are not square. Got shape {x.shape}"
                )
            x = x.reshape(-1, side, side)
        elif x.ndim == 4:
            if x.shape[1] == 1:
                x = x.squeeze(1)
            elif x.shape[-1] == 1:
                x = x.squeeze(-1)

        if x.ndim != 3:
            raise RuntimeError(
                f"[Data] Unsupported USPS image shape after normalization: {x.shape}"
            )

        self.data = x
        self.targets = y.astype(int).reshape(-1).tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.data[idx]

        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            else:
                img = img.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[idx]


def _ensure_imagefolder_split_exists(root_dir: str, split: str) -> str:
    split_dir = os.path.join(root_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"[Data] Missing ImageFolder split directory:\n"
            f"  {split_dir}\n"
            f"Expected structure:\n"
            f"  {root_dir}/{split}/0 ... 9/\n"
        )
    return split_dir


def _load_torchvision_optional_download(build_local, build_download, dataset_name: str):
    """
    Try local files first.
    Only download if local load fails AND download is allowed.
    """
    try:
        ds = build_local()
        return ds, False
    except Exception as local_e:
        if not FCL_ALLOW_DATA_DOWNLOAD:
            raise RuntimeError(
                f"[Data] Failed to load local dataset '{dataset_name}'. "
                f"Set FCL_ALLOW_DATA_DOWNLOAD=1 to download if needed.\n"
                f"Local error: {repr(local_e)}"
            ) from local_e

        try:
            ds = build_download()
            return ds, True
        except Exception as download_e:
            raise RuntimeError(
                f"[Data] Failed to load or download dataset '{dataset_name}'.\n"
                f"Local error: {repr(local_e)}\n"
                f"Download error: {repr(download_e)}"
            ) from download_e


def _load_mnist(data_root: str, train: bool):
    tf = _digit_train_transform() if train else _digit_test_transform()
    root = os.path.join(data_root, "digit5", "mnist")

    ds, used_download = _load_torchvision_optional_download(
        build_local=lambda: datasets.MNIST(
            root=root,
            train=train,
            download=False,
            transform=tf,
        ),
        build_download=lambda: datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=tf,
        ),
        dataset_name=f"MNIST ({'train' if train else 'test'})",
    )

    wrapped = _TargetsAdapterDataset(ds, list(ds.targets))
    return wrapped, used_download


def _load_svhn(data_root: str, train: bool):
    tf = _digit_rgb_train_transform() if train else _digit_rgb_test_transform()
    split = "train" if train else "test"
    root = os.path.join(data_root, "digit5", "svhn")

    ds, used_download = _load_torchvision_optional_download(
        build_local=lambda: datasets.SVHN(
            root=root,
            split=split,
            download=False,
            transform=tf,
        ),
        build_download=lambda: datasets.SVHN(
            root=root,
            split=split,
            download=True,
            transform=tf,
        ),
        dataset_name=f"SVHN ({split})",
    )

    targets = [int(y) for y in ds.labels]
    wrapped = _TargetsAdapterDataset(ds, targets)
    return wrapped, used_download


def _load_usps(data_root: str, train: bool):
    tf = _digit_train_transform() if train else _digit_test_transform()
    root = os.path.join(data_root, "digit5", "usps")
    h5_path = os.path.join(root, "usps.h5")

    if os.path.isfile(h5_path):
        ds = _USPSH5Dataset(h5_path=h5_path, train=train, transform=tf)
        return ds, False

    ds, used_download = _load_torchvision_optional_download(
        build_local=lambda: datasets.USPS(
            root=root,
            train=train,
            download=False,
            transform=tf,
        ),
        build_download=lambda: datasets.USPS(
            root=root,
            train=train,
            download=True,
            transform=tf,
        ),
        dataset_name=f"USPS ({'train' if train else 'test'})",
    )

    wrapped = _TargetsAdapterDataset(ds, list(ds.targets))
    return wrapped, used_download


def _load_mnist_m(data_root: str, train: bool):
    root_dir = os.path.join(data_root, "digit5", "mnist_m")
    split_dir = _ensure_imagefolder_split_exists(root_dir, "train" if train else "test")
    tf = _digit_rgb_train_transform() if train else _digit_rgb_test_transform()
    ds = datasets.ImageFolder(root=split_dir, transform=tf)
    return _TargetsAdapterDataset(ds, list(ds.targets)), False


def _load_syn(data_root: str, train: bool):
    root_dir = os.path.join(data_root, "digit5", "syn")
    split_dir = _ensure_imagefolder_split_exists(root_dir, "train" if train else "test")
    tf = _digit_rgb_train_transform() if train else _digit_rgb_test_transform()
    ds = datasets.ImageFolder(root=split_dir, transform=tf)
    return _TargetsAdapterDataset(ds, list(ds.targets)), False


def _digit5_domain_names() -> List[str]:
    return ["mnist", "svhn", "usps", "mnist_m", "syn"]


_DIGIT5_CACHE: Dict[Tuple[str, bool, str], Dataset] = {}


def _load_digit5_task_dataset(
    data_root: str,
    task_id: int,
    train: bool,
):
    domains = _digit5_domain_names()
    if task_id < 0 or task_id >= len(domains):
        raise IndexError(
            f"[Data] digit5 task_id={task_id} out of range. Valid range: 0..{len(domains)-1}"
        )

    domain = domains[task_id]
    cache_key = (domain, bool(train), os.path.abspath(data_root))
    if cache_key in _DIGIT5_CACHE:
        return _DIGIT5_CACHE[cache_key], domain

    if domain == "mnist":
        ds, used_download = _load_mnist(data_root, train=train)
    elif domain == "svhn":
        ds, used_download = _load_svhn(data_root, train=train)
    elif domain == "usps":
        ds, used_download = _load_usps(data_root, train=train)
    elif domain == "mnist_m":
        ds, used_download = _load_mnist_m(data_root, train=train)
    elif domain == "syn":
        ds, used_download = _load_syn(data_root, train=train)
    else:
        raise RuntimeError(f"[Data] Unknown digit5 domain: {domain}")

    if used_download:
        print(f"[Data] Downloaded digit5 domain='{domain}' into: {data_root}")
    else:
        print(f"[Data] Using local digit5 domain='{domain}' from: {data_root}")

    _DIGIT5_CACHE[cache_key] = ds
    return ds, domain


# ---------------- Task split helpers ----------------
def create_class_incremental_indices(labels: np.ndarray, num_tasks: int, classes_per_task: int):
    labels = np.array(labels)
    tasks_indices = []

    for t in range(num_tasks):
        start = t * classes_per_task
        end = start + classes_per_task
        task_classes = list(range(start, end))
        idx = np.where(np.isin(labels, task_classes))[0]
        tasks_indices.append(idx)

    return tasks_indices


def iid_split(indices: np.ndarray, num_clients: int):
    indices = np.array(indices)
    np.random.shuffle(indices)
    return np.array_split(indices, num_clients)


def dirichlet_split(
    task_labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_size: int = 5,
    max_tries: int = 30,
):
    task_labels = np.array(task_labels)
    classes = np.unique(task_labels)

    if len(task_labels) == 0:
        return [[] for _ in range(num_clients)]

    last_client_indices = [[] for _ in range(num_clients)]

    for _ in range(max_tries):
        client_indices = [[] for _ in range(num_clients)]

        for c in classes:
            idx_c = np.where(task_labels == c)[0]
            np.random.shuffle(idx_c)

            props = np.random.dirichlet(np.ones(num_clients) * alpha)
            sizes = (props * len(idx_c)).astype(int)

            diff = len(idx_c) - sizes.sum()
            if diff > 0:
                add_to = np.argsort(-props)[:diff]
                sizes[add_to] += 1
            elif diff < 0:
                for j in np.argsort(-sizes):
                    if diff == 0:
                        break
                    if sizes[j] > 0:
                        sizes[j] -= 1
                        diff += 1

            start = 0
            for client_id, sz in enumerate(sizes):
                if sz <= 0:
                    continue
                part = idx_c[start:start + sz]
                client_indices[client_id].extend(part.tolist())
                start += sz

        for i in range(num_clients):
            np.random.shuffle(client_indices[i])

        last_client_indices = client_indices
        if min(len(ci) for ci in client_indices) >= min_size:
            return client_indices

    return last_client_indices


def noniid_split_with_min_samples(
    task_idx: np.ndarray,
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples: int = 5,
    max_tries: int = 50,
):
    task_idx = np.array(task_idx)
    labels = np.array(labels)

    for _ in range(max_tries):
        task_labels = labels[task_idx]
        rel_splits = dirichlet_split(
            task_labels,
            num_clients=num_clients,
            alpha=alpha,
            min_size=min_samples,
            max_tries=30,
        )
        sizes = [len(s) for s in rel_splits]
        if min(sizes) >= min_samples:
            return [task_idx[np.array(rel, dtype=int)] for rel in rel_splits]

    print(
        f"[WARN] Dirichlet split failed to satisfy min_samples={min_samples} "
        f"after {max_tries} tries. Falling back to IID split for this task."
    )
    return iid_split(task_idx, num_clients)


def _build_client_eval_splits_from_train_indices(
    client_indices: np.ndarray,
    eval_ratio: float = 0.2,
    min_eval_samples: int = 1,
):
    """
    Split a client's task-specific TRAIN indices into:
      - train subset for optimization
      - eval subset for local fairness evaluation fallback
    This creates a local held-out split when dataset-level client test partitions
    are not available.
    """
    client_indices = np.array(client_indices, dtype=int)
    if client_indices.size == 0:
        return client_indices, client_indices

    shuffled = client_indices.copy()
    np.random.shuffle(shuffled)

    eval_size = int(round(len(shuffled) * eval_ratio))
    eval_size = max(min_eval_samples, eval_size)
    if len(shuffled) >= 2:
        eval_size = min(eval_size, len(shuffled) - 1)
    else:
        eval_size = 1

    eval_idx = shuffled[:eval_size]
    train_idx = shuffled[eval_size:]

    if train_idx.size == 0:
        train_idx = shuffled[:1]
        eval_idx = shuffled[1:] if len(shuffled) > 1 else shuffled[:1]

    if eval_idx.size == 0:
        eval_idx = train_idx[:1]

    return train_idx, eval_idx


# ---------------- Main: client train/test loaders ----------------
def get_clients_data(
    num_clients: int = 10,
    num_tasks: int = 5,
    classes_per_task: int = 2,
    alpha: float = 0.3,
    iid: bool = True,
    dataset_name: str = "cifar10",
    data_root: Optional[str] = None,
    batch_size: int = 32,
    min_samples_per_client_per_task: int = 5,
    max_tries: int = 50,
):
    """
    Returns:
      train_clients_data: List[List[DataLoader]]
      test_clients_data:  List[List[DataLoader]]

    For CIFAR / Tiny-ImageNet / Digit5, this function builds per-client local train loaders and
    per-client local evaluation loaders. The local eval loaders are constructed as:

      - CIFAR: held-out split from each client's task-specific train partition
      - Tiny-ImageNet: held-out split from each client's task-specific train partition
      - Digit5: held-out split from each client's task-specific train partition
    """
    data_root = _normalize_data_root(data_root)
    _ensure_data_root_accessible(data_root)

    dn = dataset_name.lower()

    # -------- digit5 domain-incremental --------
    if dn == "digit5":
        max_digit_tasks = 5
        if num_tasks > max_digit_tasks:
            raise ValueError(f"[Data] digit5 supports at most {max_digit_tasks} tasks, got {num_tasks}")

        train_clients_data: List[List[DataLoader]] = [[] for _ in range(num_clients)]
        test_clients_data: List[List[DataLoader]] = [[] for _ in range(num_clients)]

        for task_id in range(num_tasks):
            train_ds, domain_name = _load_digit5_task_dataset(
                data_root=data_root,
                task_id=task_id,
                train=True,
            )

            labels = np.array(train_ds.targets)
            all_idx = np.arange(len(train_ds), dtype=int)

            if iid:
                splits = iid_split(all_idx, num_clients)
            else:
                splits = noniid_split_with_min_samples(
                    task_idx=all_idx,
                    labels=labels,
                    num_clients=num_clients,
                    alpha=alpha,
                    min_samples=min_samples_per_client_per_task,
                    max_tries=max_tries,
                )

            for cid in range(num_clients):
                client_indices = np.array(splits[cid], dtype=int)

                if client_indices.size == 0:
                    raise RuntimeError(
                        f"[Data] Empty subset detected for digit5. task={task_id} ({domain_name}), client={cid}."
                    )

                train_idx, eval_idx = _build_client_eval_splits_from_train_indices(
                    client_indices=client_indices,
                    eval_ratio=0.2,
                    min_eval_samples=1,
                )

                train_subset = Subset(train_ds, train_idx.tolist())
                eval_subset = Subset(train_ds, eval_idx.tolist())

                train_loader = DataLoader(
                    train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=False,
                )
                eval_loader = DataLoader(
                    eval_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )

                train_clients_data[cid].append(train_loader)
                test_clients_data[cid].append(eval_loader)

        for cid in range(num_clients):
            if len(train_clients_data[cid]) != num_tasks:
                raise RuntimeError(
                    f"[Data] Client {cid} has {len(train_clients_data[cid])} train task loaders, expected {num_tasks}"
                )
            if len(test_clients_data[cid]) != num_tasks:
                raise RuntimeError(
                    f"[Data] Client {cid} has {len(test_clients_data[cid])} test task loaders, expected {num_tasks}"
                )

        print(f"[Data] Built digit5 client train/test loaders from: {data_root}")
        return train_clients_data, test_clients_data

    # -------- CIFAR / Tiny-ImageNet class-incremental --------
    train_tf, _ = get_transforms(dataset_name)

    if dn in ("cifar10", "cifar100"):
        train_ds, used_download = _load_cifar(dataset_name, data_root, train=True, transform=train_tf)
        if used_download:
            print(f"[Data] Downloaded {dataset_name} train set into: {data_root}")
        else:
            print(f"[Data] Using existing {dataset_name} train set at: {data_root}")

    elif dn == "tinyimagenet":
        train_ds, _ = _load_tinyimagenet(data_root, train=True, transform=train_tf)
        print(f"[Data] Using existing tinyimagenet train set at: {data_root}")

    else:
        raise ValueError(f"[Data] Unsupported dataset_name='{dataset_name}'")

    labels = np.array(train_ds.targets)
    total_classes = get_dataset_num_classes(dataset_name)

    if num_tasks * classes_per_task > total_classes:
        raise ValueError(
            f"Invalid split: num_tasks({num_tasks}) * classes_per_task({classes_per_task}) "
            f"> total_classes({total_classes})"
        )

    tasks_indices = create_class_incremental_indices(labels, num_tasks, classes_per_task)
    train_clients_data: List[List[DataLoader]] = [[] for _ in range(num_clients)]
    test_clients_data: List[List[DataLoader]] = [[] for _ in range(num_clients)]

    for task_id, task_idx in enumerate(tasks_indices):
        task_idx = np.array(task_idx)

        if iid:
            splits = iid_split(task_idx, num_clients)
        else:
            splits = noniid_split_with_min_samples(
                task_idx=task_idx,
                labels=labels,
                num_clients=num_clients,
                alpha=alpha,
                min_samples=min_samples_per_client_per_task,
                max_tries=max_tries,
            )

        for cid in range(num_clients):
            client_indices = np.array(splits[cid], dtype=int)

            if client_indices.size == 0:
                raise RuntimeError(
                    f"[Data] Empty subset detected. task={task_id}, client={cid}. "
                    f"Increase min_samples_per_client_per_task or alpha."
                )

            train_idx, eval_idx = _build_client_eval_splits_from_train_indices(
                client_indices=client_indices,
                eval_ratio=0.2,
                min_eval_samples=1,
            )

            train_subset = Subset(train_ds, train_idx.tolist())
            eval_subset = Subset(train_ds, eval_idx.tolist())

            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            eval_loader = DataLoader(
                eval_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )

            train_clients_data[cid].append(train_loader)
            test_clients_data[cid].append(eval_loader)

    if len(train_clients_data) != num_clients:
        raise RuntimeError(
            f"[Data] Expected {num_clients} clients, built {len(train_clients_data)}"
        )

    for cid in range(num_clients):
        if len(train_clients_data[cid]) != num_tasks:
            raise RuntimeError(
                f"[Data] Client {cid} has {len(train_clients_data[cid])} train task loaders, expected {num_tasks}"
            )
        if len(test_clients_data[cid]) != num_tasks:
            raise RuntimeError(
                f"[Data] Client {cid} has {len(test_clients_data[cid])} test task loaders, expected {num_tasks}"
            )

    print(f"[Data] Built {dataset_name} client train/test loaders from: {data_root}")
    return train_clients_data, test_clients_data


# ---------------- Task-aware TEST loader ----------------
def get_task_test_loader(
    dataset_name: str = "cifar10",
    data_root: Optional[str] = None,
    task_id: int = 0,
    classes_per_task: int = 2,
    batch_size: int = 128,
):
    data_root = _normalize_data_root(data_root)
    _ensure_data_root_accessible(data_root)

    dn = dataset_name.lower()

    # -------- digit5 domain-incremental --------
    if dn == "digit5":
        test_ds, domain_name = _load_digit5_task_dataset(
            data_root=data_root,
            task_id=task_id,
            train=False,
        )

        loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        task_classes = list(range(10))
        print(f"[Data] Using digit5 test domain='{domain_name}' from: {data_root}")
        return loader, task_classes

    # -------- CIFAR / Tiny-ImageNet class-incremental --------
    _, test_tf = get_transforms(dataset_name)

    if dn in ("cifar10", "cifar100"):
        test_ds, used_download = _load_cifar(dataset_name, data_root, train=False, transform=test_tf)

        if used_download:
            print(f"[Data] Downloaded {dataset_name} test set into: {data_root}")
        else:
            print(f"[Data] Using existing {dataset_name} test set at: {data_root}")

    elif dn == "tinyimagenet":
        test_ds, _ = _load_tinyimagenet(data_root, train=False, transform=test_tf)
        print(f"[Data] Using existing tinyimagenet val set at: {data_root}")

    else:
        raise ValueError(f"[Data] Unsupported dataset_name='{dataset_name}'")

    targets = np.array(test_ds.targets)
    total_classes = get_dataset_num_classes(dataset_name)

    start_class = task_id * classes_per_task
    end_class = start_class + classes_per_task
    if end_class > total_classes:
        raise ValueError(
            f"Task classes exceed total classes: end_class={end_class}, total_classes={total_classes}"
        )

    task_classes = list(range(start_class, end_class))
    idx = np.where(np.isin(targets, task_classes))[0]

    if len(idx) == 0:
        raise RuntimeError(
            f"[Data] No test samples found for task_id={task_id}, classes={task_classes}"
        )

    subset = Subset(test_ds, idx.tolist())
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader, task_classes