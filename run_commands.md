# Representative Experiment Commands

## CIFAR-10

python3 -m src.main --dataset cifar10 --method TASK_ADAPTER --fl_algo FEDAVG --alpha 0.7 --iid 0 --seeds 42 43 44

## CIFAR-100

python3 -m src.main --dataset cifar100 --method ALL --fl_algo FEDAVG --alpha 0.7 --iid 0 --seeds 42 43 44

## Digit5

python3 -m src.main --dataset digit5 --method ALL --fl_algo FEDAVG --alpha 0.7 --iid 0 --seeds 42 43 44

## Tiny-ImageNet

python3 -m src.main --dataset tinyimagenet --method ALL --fl_algo FEDAVG --alpha 0.7 --iid 0 --seeds 42 43 44

## Aggregation

python3 src/aggregate_results.py --root experiments/latest/run_YYYYMMDD_HHMMSS --out_dir experiments/latest/run_YYYYMMDD_HHMMSS/aggregated
