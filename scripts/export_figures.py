"""Export key figures from notebooks to docs/documentation/images/results/.

Generates the static PNG files embedded in the final documentation PDF
(chapter 6 — wyniki eksperymentów). Re-runs are idempotent: existing files
are overwritten with the latest data from ``results/``.

Usage::

    python scripts/export_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "notebooks"))
import _helpers as H  # noqa: E402

OUT_DIR = REPO / "docs" / "documentation" / "images" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


# ---------------------------------------------------------------------------
# 6.1 quality
# ---------------------------------------------------------------------------


def fig_quality_bars() -> None:
    comp = H.load_comparison().set_index("dataset")
    order = sorted(H.DATASETS_WITH_GT,
                   key=lambda d: -(comp.loc[d, "ari"] + comp.loc[d, "nmi"]) / 2)
    ari = comp.loc[order, "ari"]
    nmi = comp.loc[order, "nmi"]
    quality = (ari + nmi) / 2

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(order))
    w = 0.27
    ax.bar(x - w, ari, w, label="ARI", color="#4C72B0")
    ax.bar(x, nmi, w, label="NMI", color="#DD8452")
    ax.bar(x + w, quality, w, label="Quality = (ARI+NMI)/2", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels([H.PRETTY[d] for d in order])
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls=":", color="gray", linewidth=0.8)
    ax.set_ylabel("score")
    ax.set_title("Jakość pychameleona na datasetach z ground-truth")
    ax.legend(loc="lower left", framealpha=0.9)
    for i, (a, n, q) in enumerate(zip(ari, nmi, quality)):
        ax.text(i - w, a + 0.015, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i, n + 0.015, f"{n:.2f}", ha="center", fontsize=8)
        ax.text(i + w, q + 0.015, f"{q:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quality_bars.png")
    plt.close(fig)


def fig_quality_gallery() -> None:
    comp = H.load_comparison().set_index("dataset")
    order = sorted(H.DATASETS_WITH_GT,
                   key=lambda d: -(comp.loc[d, "ari"] + comp.loc[d, "nmi"]) / 2)
    fig, axes = plt.subplots(len(order), 3, figsize=(11, 3.5 * len(order)))
    for i, name in enumerate(order):
        gt_df = H.load_ground_truth(name)
        pred_df = H.load_pychameleon_labels(name)
        pred_aligned = H.align_labels_to_gt(pred_df["label"].values, gt_df["label"].values)
        pred_view = pred_df.copy()
        pred_view["label"] = pred_aligned

        H.scatter(axes[i, 0], gt_df, title=f"{H.PRETTY[name]} — ground truth")
        H.scatter(axes[i, 1], pred_view, title=f"{H.PRETTY[name]} — pychameleon")
        nc, ni, _ = H.correctness_scatter(
            axes[i, 2], pred_df[["x0", "x1"]], gt_df["label"].values,
            pred_aligned, title="",
        )
        total = nc + ni
        pct = 100 * nc / total if total else 0
        axes[i, 2].set_title(f"{H.PRETTY[name]} — accuracy {pct:.1f}%", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "quality_gallery.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.2 vs Moonpuck
# ---------------------------------------------------------------------------


def fig_moonpuck_speedup() -> None:
    comp = H.load_comparison().set_index("dataset")
    rows = []
    for d in H.DATASETS_WITH_MOONPUCK:
        if d not in comp.index:
            continue
        py_rt = comp.loc[d, "runtime_s"]
        mp_rt = comp.loc[d, "runtime_moonpuck_s"]
        if pd.isna(mp_rt) or pd.isna(py_rt) or py_rt == 0:
            continue
        rows.append((d, comp.loc[d, "n"], py_rt, mp_rt, mp_rt / py_rt))
    df = pd.DataFrame(rows, columns=["dataset", "n", "py_s", "mp_s", "speedup"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["n"], df["speedup"], s=70, color="#4C72B0", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(
            f"{H.PRETTY[r['dataset']]}\n{r['speedup']:.0f}×",
            (r["n"], r["speedup"]),
            xytext=(10, -4), textcoords="offset points", fontsize=9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rozmiar zbioru n")
    ax.set_ylabel("speedup pychameleon vs Moonpuck (×)")
    ax.set_title("Speedup vs implementacja referencyjna Moonpuck")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "moonpuck_speedup.png")
    plt.close(fig)


def fig_moonpuck_gallery() -> None:
    """3 columns × 2 rows: aggregation + t4_8k, each showing GT/pychameleon/Moonpuck."""
    datasets = ["aggregation", "t4_8k"]
    fig, axes = plt.subplots(len(datasets), 3, figsize=(11, 3.5 * len(datasets)))
    for i, name in enumerate(datasets):
        gt_df = H.load_ground_truth(name)
        py_df = H.load_pychameleon_labels(name)
        mp_df = H.load_moonpuck_labels(name)

        if gt_df is not None:
            H.scatter(axes[i, 0], gt_df, title=f"{H.PRETTY[name]} — ground truth")
        else:
            axes[i, 0].set_visible(False)

        H.scatter(axes[i, 1], py_df, title=f"{H.PRETTY[name]} — pychameleon")

        if mp_df is not None:
            H.scatter(axes[i, 2], mp_df, title=f"{H.PRETTY[name]} — Moonpuck")
        else:
            axes[i, 2].set_visible(False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "moonpuck_gallery.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.3 sensitivity
# ---------------------------------------------------------------------------


def _sweep_panel(ax, df: pd.DataFrame, metric: str, title: str) -> None:
    agg = df.groupby("value")[metric].agg(["mean", "std"]).reset_index()
    ax.plot(agg["value"], agg["mean"], marker="o", color="#4C72B0")
    ax.fill_between(
        agg["value"],
        agg["mean"] - agg["std"].fillna(0),
        agg["mean"] + agg["std"].fillna(0),
        alpha=0.2, color="#4C72B0",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("wartość parametru")
    ax.set_ylabel(metric.upper())
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_ylim(0, 1.05)


def fig_sensitivity(param: str) -> None:
    df = H.load_sweep(param)
    df = df[df["dataset"].isin(H.DATASETS_WITH_GT)].copy()
    df["value"] = pd.to_numeric(df["value"])
    df["quality"] = (df["ari"] + df["nmi"]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    _sweep_panel(axes[0], df, "ari", f"ARI vs {param}")
    _sweep_panel(axes[1], df, "nmi", f"NMI vs {param}")
    _sweep_panel(axes[2], df, "quality", f"Quality vs {param}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"sensitivity_{param}.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6.4 scalability
# ---------------------------------------------------------------------------


def fig_scalability_n() -> None:
    df = H.load_scalability()
    df_n = df[df["kind"] == "n"].copy()
    agg = df_n.groupby("n")["runtime_s"].agg(["mean", "std"]).reset_index()

    log_n = np.log(agg["n"].values)
    log_t = np.log(agg["mean"].values)
    slope, intercept = np.polyfit(log_n, log_t, 1)
    fit_t = np.exp(intercept) * agg["n"] ** slope

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(agg["n"], agg["mean"], yerr=agg["std"].fillna(0),
                fmt="o", color="#4C72B0", label="pychameleon (mean ± std, 3 repeats)")
    ax.plot(agg["n"], fit_t, ls="--", color="#DD8452",
            label=f"fit: O(n^{slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rozmiar zbioru n")
    ax.set_ylabel("runtime [s]")
    ax.set_title("Skalowalność względem liczby punktów (d = 2)")
    ax.legend()
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scalability_n.png")
    plt.close(fig)


def fig_scalability_d() -> None:
    df = H.load_scalability()
    df_d = df[df["kind"] == "d"].copy()
    if df_d.empty:
        return
    agg = df_d.groupby("d")["runtime_s"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(agg["d"], agg["mean"], yerr=agg["std"].fillna(0),
                fmt="o-", color="#4C72B0", label="pychameleon (mean ± std)")
    ax.set_xlabel("liczba wymiarów d (n = 2000)")
    ax.set_ylabel("runtime [s]")
    ax.set_title("Skalowalność względem wymiarowości")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scalability_d.png")
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> None:
    print(f"Eksport figur do {OUT_DIR}")
    fig_quality_bars()
    print("  + quality_bars.png")
    fig_quality_gallery()
    print("  + quality_gallery.png")
    fig_moonpuck_speedup()
    print("  + moonpuck_speedup.png")
    fig_moonpuck_gallery()
    print("  + moonpuck_gallery.png")
    for p in ("k_nn", "alpha", "min_cluster_size"):
        fig_sensitivity(p)
        print(f"  + sensitivity_{p}.png")
    fig_scalability_n()
    print("  + scalability_n.png")
    fig_scalability_d()
    print("  + scalability_d.png")
    print("gotowe.")


if __name__ == "__main__":
    main()
