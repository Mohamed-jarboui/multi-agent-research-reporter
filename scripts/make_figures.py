"""Regenerate docs/figs/*.svg from docs/progress.json and results/*.json.

Needs the `report` dependency group:  uv run --group report python scripts/make_figures.py

Figures (a figure is skipped, with a message, when its input is missing):
  stage_progress.svg   status of R0-R4 from docs/progress.json
  r1_worker_timeline.svg  Gantt of worker attempts per topic, from results/r1_timeline.json:
                       {"runs": [{"topic": str, "wall_s": float, "workers": [{"section": str,
                                  "start_s": float, "end_s": float, "status": str}, ...]}]}
  baseline_smoke.svg   tokens, cost and latency per topic, from results/baseline_smoke.json:
                       {"runs": [{"topic": str, "total_tokens": int, "cost_usd": float,
                                  "latency_s": float}, ...]}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figs"
STATUS_LEVEL = {"todo": 0, "in_progress": 1, "pr_open": 2, "merged": 3}
STATUS_COLOUR = {
    "todo": "#c9ced6",
    "in_progress": "#7fa8e6",
    "pr_open": "#e9a23b",
    "merged": "#2f7d5b",
}
INK = "#1f2933"
BAR = "#3b6fb6"


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=INK)
    ax.title.set_color(INK)


def stage_progress() -> Path | None:
    path = ROOT / "docs" / "progress.json"
    if not path.is_file():
        print("skip stage_progress: docs/progress.json not found")
        return None
    stages = json.loads(path.read_text(encoding="utf-8"))["stages"]
    fig, ax = plt.subplots(figsize=(7, 1.2 + 0.4 * len(stages)))
    labels = [f"{s['id']}  {s['name']}" for s in stages][::-1]
    levels = [STATUS_LEVEL[s["status"]] + 0.5 for s in stages][::-1]
    colours = [STATUS_COLOUR[s["status"]] for s in stages][::-1]
    ax.barh(labels, levels, color=colours)
    ax.set_xlim(0, 4)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], ["todo", "in progress", "PR open", "merged"])
    ax.set_title("Stage progress")
    _style(ax)
    return _save(fig, "stage_progress.svg")


def baseline_smoke() -> Path | None:
    path = ROOT / "results" / "baseline_smoke.json"
    if not path.is_file():
        print("skip baseline_smoke: results/baseline_smoke.json not measured yet")
        return None
    runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    if not runs:
        print("skip baseline_smoke: no runs in results/baseline_smoke.json")
        return None
    topics = [r["topic"][:32] for r in runs]
    panels = [
        ("Tokens", "total_tokens", "{:,.0f}"),
        ("Cost (USD)", "cost_usd", "${:.4f}"),
        ("Latency (s)", "latency_s", "{:.0f}"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 1.5 + 0.5 * len(runs)), sharey=True)
    for ax, (title, key, fmt) in zip(axes, panels, strict=True):
        values = [r[key] for r in runs]
        ax.barh(topics[::-1], values[::-1], color=BAR)
        for y, v in enumerate(values[::-1]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_title(title)
        _style(ax)
    fig.suptitle(f"Baseline smoke runs (n={len(runs)})", color=INK)
    return _save(fig, "baseline_smoke.svg")


def r1_worker_timeline() -> Path | None:
    path = ROOT / "results" / "r1_timeline.json"
    if not path.is_file():
        print("skip r1_worker_timeline: results/r1_timeline.json not measured yet")
        return None
    runs = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
    if not runs:
        print("skip r1_worker_timeline: no runs in results/r1_timeline.json")
        return None
    colours = {"complete": "#2f7d5b", "incomplete": "#b23a3a"}
    height = sum(0.45 * len(r["workers"]) + 1.0 for r in runs)
    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(9, height),
        gridspec_kw={"height_ratios": [len(r["workers"]) + 2 for r in runs]},
    )
    if len(runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, runs, strict=True):
        workers = run["workers"]
        for i, w in enumerate(workers):
            retry = "attempt 2" in w["section"]
            ax.barh(
                i,
                w["end_s"] - w["start_s"],
                left=w["start_s"],
                color=colours.get(w["status"], "#888888"),
                hatch="//" if retry else None,
                edgecolor="white" if retry else colours.get(w["status"], "#888888"),
                height=0.6,
            )
            ax.text(
                w["end_s"] + 2,
                i,
                f"{w['end_s'] - w['start_s']:.0f} s",
                va="center",
                fontsize=8,
                color=INK,
            )
        ax.set_yticks(range(len(workers)), [w["section"] for w in workers])
        ax.invert_yaxis()
        ax.set_xlim(0, run["wall_s"] * 1.1)
        ax.set_xlabel("seconds since run start", color=INK)
        ax.set_title(f"{run['topic']} (wall {run['wall_s']:.0f} s, n=1)", fontsize=10)
        _style(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours["complete"]),
        plt.Rectangle((0, 0), 1, 1, color=colours["incomplete"]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colours["incomplete"], hatch="//", edgecolor="white"),
    ]
    fig.legend(
        handles,
        ["complete", "incomplete", "retry attempt (hatched)"],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("R1 worker attempts over time", color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, "r1_worker_timeline.svg")


def _save(fig, name: str) -> Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / name
    if not fig.get_tight_layout():
        fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")
    return out


def main() -> None:
    stage_progress()
    baseline_smoke()
    r1_worker_timeline()


if __name__ == "__main__":
    main()
