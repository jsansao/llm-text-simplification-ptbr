import json, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

MODELS = [
    ("PTT5 LoRA r=8", "models/ptt5-small-ft/checkpoint-1340/trainer_state.json"),
    ("PTT5 LoRA r=32", "models/ptt5-small-ft-rank32/checkpoint-1230/trainer_state.json"),
    ("PTT5 full FT", "models/ptt5-small-ft-full/checkpoint-1230/trainer_state.json"),
    ("Qwen 0.5B QLoRA", "models/qwen25-05b-ft/checkpoint-1225/trainer_state.json"),
]

def load_losses(path):
    with open(path) as f:
        state = json.load(f)
    train = [(e["step"], e["loss"]) for e in state["log_history"] if "loss" in e]
    eval_ = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]
    return train, eval_

fig, axes = plt.subplots(
    1, 4, figsize=(10.0, 2.4), sharey=False,
    gridspec_kw={"wspace": 0.30},
)

for ax, (label, path) in zip(axes, MODELS):
    train, eval_ = load_losses(BASE / path)
    steps_train, losses = zip(*train) if train else ([], [])
    steps_eval, evals = zip(*eval_) if eval_ else ([], [])
    ax.plot(steps_train, losses, color="tab:blue", lw=1.2, label="Treino")
    ax.plot(steps_eval, evals, color="tab:orange", marker="o", ls="", ms=4, label="Validação")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(label, fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

fig.savefig(BASE / "artigo/figs/loss_curves.pdf", bbox_inches="tight", pad_inches=0.02)
print("OK: loss_curves.pdf gerado")
