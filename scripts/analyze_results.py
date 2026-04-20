"""Parse results.csv and print a clean model comparison + best-config table."""
import os
import sys
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results.csv")

df = pd.read_csv(CSV_PATH)

# Keep only sweep runs (have best_val_acc logged)
df = df[df["best_val_acc"].notna() & df["model"].notna()].copy()
df["best_val_acc"] = df["best_val_acc"].astype(float)
df["best_val_f1_macro"] = df["best_val_f1_macro"].astype(float)
df["epoch_time_s"] = df["epoch_time_s"].astype(float)

# ── Per-model best run ────────────────────────────────────────────────
idx_best = df.groupby("model")["best_val_acc"].idxmax()
best = df.loc[idx_best].set_index("model")

MODELS = ["EfficientNetV2-S", "ConvNeXt-Small", "Swin-T"]

print("\n" + "="*70)
print("  BEST RUN PER MODEL")
print("="*70)
for m in MODELS:
    if m not in best.index:
        continue
    r = best.loc[m]
    print(f"\n{m}")
    print(f"  val_acc       : {r['best_val_acc']:.4f}  ({r['best_val_acc']*100:.2f}%)")
    print(f"  val_f1_macro  : {r['best_val_f1_macro']:.4f}")
    print(f"  optimizer     : {r['optimizer']}")
    print(f"  lr_head       : {r['lr_head']:.2e}")
    print(f"  lr_backbone   : {r['lr_backbone']:.2e}")
    print(f"  weight_decay  : {r['weight_decay']}")
    print(f"  unfreeze_layers: {int(r['unfreeze_layers'])}")
    print(f"  warmup_epochs : {int(r['warmup_epochs'])}")
    print(f"  finetune_epochs: {int(r['finetune_epochs'])}")
    print(f"  epoch_time_s  : {r['epoch_time_s']:.1f} s")
    print(f"  run_name      : {r['Name']}")

# ── Summary table ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("  COMPARISON TABLE")
print("="*70)
print(f"{'Model':<22} {'Val Acc':>9} {'F1 Macro':>10} {'s/epoch':>9} {'Optimizer':<10} {'lr_head':>9} {'unfreeze':>9}")
print("-"*70)
for m in MODELS:
    if m not in best.index:
        continue
    r = best.loc[m]
    print(f"{m:<22} {r['best_val_acc']*100:>8.2f}% {r['best_val_f1_macro']*100:>9.2f}%"
          f" {r['epoch_time_s']:>8.1f}s {r['optimizer']:<10} {r['lr_head']:>9.2e}"
          f" {int(r['unfreeze_layers']):>8}")

# ── All runs per model (sorted) ───────────────────────────────────────
print("\n" + "="*70)
print("  ALL RUNS SUMMARY (sorted by val_acc)")
print("="*70)
for m in MODELS:
    sub = df[df["model"] == m].sort_values("best_val_acc", ascending=False)
    print(f"\n{m}  ({len(sub)} runs)")
    print(f"  {'Run name':<52} {'val_acc':>8} {'f1':>7}")
    print("  " + "-"*68)
    for _, r in sub.iterrows():
        print(f"  {r['Name']:<52} {r['best_val_acc']*100:>7.2f}% {r['best_val_f1_macro']*100:>6.2f}%")
