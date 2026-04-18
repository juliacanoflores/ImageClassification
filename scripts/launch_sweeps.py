"""Create and launch W&B hyperparameter sweeps for all three models.

Each sweep uses Bayesian optimisation over:
  - lr_head / lr_backbone (log-uniform)
  - optimizer (AdamW, SGD)
  - weight_decay
  - unfreeze_layers (0–3 backbone blocks)
  - warmup_epochs / finetune_epochs

Usage:
  # Create sweeps + run 12 runs per model (36 total)
  python scripts/launch_sweeps.py

  # Custom count or single model
  python scripts/launch_sweeps.py --count 15
  python scripts/launch_sweeps.py --model Swin-T --count 12

  # Only create sweeps (get IDs), run agents later / on another machine
  python scripts/launch_sweeps.py --create-only

  # Resume existing sweep by ID
  python scripts/launch_sweeps.py --sweep-id <id> --model <name> --count 12
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb

from sweep_train import ENTITY, MODEL_REGISTRY, PROJECT, train_run

SWEEP_BASE = {
    "method": "bayes",
    "metric": {"name": "best_val_acc", "goal": "maximize"},
    "parameters": {
        "lr_head": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-2,
        },
        "lr_backbone": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4,
        },
        "optimizer": {"values": ["AdamW", "SGD"]},
        "weight_decay": {"values": [0.001, 0.01, 0.05, 0.1]},
        "unfreeze_layers": {"values": [0, 1, 2, 3]},
        "warmup_epochs": {"values": [3, 5]},
        "finetune_epochs": {"values": [5, 8, 10]},
    },
}


def create_sweep(model_name: str) -> str:
    config = {**SWEEP_BASE, "name": f"sweep-{model_name}"}
    config["parameters"] = {**SWEEP_BASE["parameters"], "model": {"value": model_name}}
    sweep_id = wandb.sweep(config, entity=ENTITY, project=PROJECT)
    print(f"[{model_name}] sweep created → {sweep_id}")
    return sweep_id


def run_agent(sweep_id: str, count: int) -> None:
    wandb.agent(sweep_id, function=train_run, count=count, entity=ENTITY, project=PROJECT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch W&B sweeps for image classification")
    parser.add_argument("--count", type=int, default=12, help="Runs per model (default 12)")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Run a single model instead of all three")
    parser.add_argument("--create-only", action="store_true",
                        help="Create sweeps but do not start agents")
    parser.add_argument("--sweep-id", type=str, default=None,
                        help="Resume an existing sweep instead of creating a new one")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_REGISTRY.keys())

    if args.sweep_id:
        if len(models) != 1:
            parser.error("--sweep-id requires exactly one --model")
        print(f"Resuming sweep {args.sweep_id} for {models[0]} ({args.count} runs)")
        run_agent(args.sweep_id, args.count)
        return

    for model_name in models:
        sweep_id = create_sweep(model_name)
        if not args.create_only:
            print(f"[{model_name}] starting agent for {args.count} runs …")
            run_agent(sweep_id, args.count)


if __name__ == "__main__":
    main()
