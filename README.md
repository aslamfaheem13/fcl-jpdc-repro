# Edge-Oriented Federated Continual Learning on Embedded Hardware

This repository contains the reproducibility package for the JPDC manuscript:

Federated Continual Learning on Embedded Hardware: An Empirical Study with Parameter-Efficient Adapters

## Overview

This project evaluates federated continual learning (FCL) on NVIDIA Jetson AGX Xavier under non-IID sequential learning conditions. It compares full-model federated training with parameter-efficient update strategies, including head-only updates, shared adapters, and task-specific adapters.

## Main methods

- HEAD_ONLY
- TRUE_FEDAVG
- SHARED_ADAPTER
- TASK_ADAPTER

## Datasets

The experiments use publicly available datasets:

- CIFAR-10
- CIFAR-100
- Digit5
- Tiny-ImageNet

Datasets are not included in this repository because of size and license considerations. Please download them from their original sources.

## Repository structure

src/              Core implementation files
configs/          Example experiment configurations
results_summary/  Processed result summaries used in the paper
figures/          Main manuscript figures and plots
docs/             Additional notes

## Reproducibility note

Large raw logs, trained model checkpoints, downloaded datasets, and virtual environments are not included. They are available from the corresponding author upon reasonable request, subject to storage and sharing limitations.

## Citation

If you use this code, please cite the corresponding JPDC paper after publication.
