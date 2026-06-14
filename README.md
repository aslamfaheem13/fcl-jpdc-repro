# Edge-Oriented Federated Continual Learning on Embedded Hardware

This repository contains the reproducibility package for the JPDC manuscript:

**Federated Continual Learning on Embedded Hardware: An Empirical Study with Parameter-Efficient Adapters**

## Overview

This project evaluates federated continual learning (FCL) on embedded edge hardware under non-IID sequential learning conditions. The experiments are designed to study the trade-off among plasticity, stability, communication efficiency, runtime, and embedded deployment behavior.

The study compares full-model federated continual learning with parameter-efficient update strategies on an NVIDIA Jetson AGX Xavier platform. The evaluated strategies include head-only updates, shared adapters, task-specific adapters, and full-model federated training.

## Evaluated Methods

The repository supports the following main update strategies:

* `HEAD_ONLY`: updates only the task-specific classifier head.
* `TRUE_FEDAVG`: updates the full model using federated averaging.
* `SHARED_ADAPTER`: updates a shared adapter module and classifier heads.
* `TASK_ADAPTER`: updates task-specific adapters and classifier heads.

## Datasets

The experiments use publicly available datasets:

* CIFAR-10
* CIFAR-100
* Digit5
* Tiny-ImageNet

Datasets are not included in this repository because of size and license considerations. Please download them from their original sources and place them according to the expected data directory structure used by the code.

## Repository Structure

```text
src/              Core implementation files for the FCL framework
configs/          Reserved for representative experiment configurations
results_summary/  Processed result summaries used in the manuscript
figures/          Main result figures and supporting plots
docs/             Additional notes for reproducibility
```

Important source files include:

```text
src/main.py               Main experiment entry point
src/client.py             Client-side local training logic
src/server.py             Federated aggregation and evaluation logic
src/data.py               Dataset loading and task construction
src/model.py              Backbone, adapter, and classifier definitions
src/config.py             Experiment configuration utilities
src/aggregate_results.py  Result aggregation and summary generation
```

## Representative Commands

Representative experiment commands are provided in:

```text
run_commands.md
```

Example:

```bash
python3 -m src.main --dataset cifar100 --method ALL --fl_algo FEDAVG --alpha 0.7 --iid 0 --seeds 42 43 44
```

Processed summaries and selected figures are provided to support the main results reported in the manuscript.

## Hardware and Software Environment

The experiments were conducted on an NVIDIA Jetson AGX Xavier embedded platform. Additional environment information is provided in:

```text
environment.md
```

The implementation uses Python and PyTorch. Hardware monitoring was performed where available using Jetson platform tools such as `tegrastats`.

## Reproducibility Note

This repository does not include:

* downloaded datasets
* virtual environments
* trained model checkpoints
* large raw experiment logs
* temporary experiment folders
* private system files

These files are excluded because of file-size limitations and storage considerations. Large raw logs and trained checkpoints are available from the corresponding author upon reasonable request.

## Processed Results

The folder `results_summary/` contains processed CSV files used to summarize the experimental results. These files are intended to help readers verify the reported trends without requiring the full raw experiment directory.

The folder `figures/` contains selected result figures and supporting plots used for manuscript preparation.

## Data and Code Availability

The source code, representative run commands, selected figures, and processed result summaries are available in this repository. Public datasets should be obtained from their original sources.

## Citation

If you use this repository or build on this work, please cite the corresponding JPDC paper after publication.
