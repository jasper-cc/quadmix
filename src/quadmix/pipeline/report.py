"""QuaDMix report generator — produces MD report + companion PNG files.

Figures saved as PNG alongside the report, referenced via standard
markdown ![](filename.png) syntax — supported by all MD viewers.
"""

import json
import os
import time
from typing import List, Tuple

import unicodedata

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quadmix.core.types import ParameterSet

from quadmix.constants import DOMAIN_SHORT_NAMES


_CJK_FONT_AVAILABLE = False


def _str_has_cjk(s):
    return any(unicodedata.category(c) == 'Lo' for c in str(s))


def _get_domain_short(num_domains, domain_names=None):
    if domain_names is not None:
        names = [str(n) for n in domain_names[:num_domains]]
        if not _CJK_FONT_AVAILABLE and any(_str_has_cjk(n) for n in names):
            names = [f"D{i}" for i in range(len(names))]
        if num_domains > len(domain_names):
            names += [f"D{i}" for i in range(len(domain_names), num_domains)]
        return names
    if num_domains <= len(DOMAIN_SHORT_NAMES):
        return DOMAIN_SHORT_NAMES[:num_domains]
    return DOMAIN_SHORT_NAMES + [f"D{i}" for i in range(len(DOMAIN_SHORT_NAMES), num_domains)]

_DEFAULT_QUALITY_NAMES = ["DCLM", "FineWeb-Edu", "English", "Math (Gen)", "Math (OpenWeb)"]
_DEFAULT_QUALITY_SHORT = ["DCLM", "Edu", "Eng", "MathG", "MathO"]

COLOR_ORIG = "#5B9BD5"
COLOR_OPT = "#ED7D31"
QUALITY_COLORS = ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5",
                   "#70AD47", "#264478", "#9B59B6", "#2ECC71", "#E74C3C"]


_CJK_TEST_CHAR = '\u6570'


def _find_cjk_font():
    fm = matplotlib.font_manager
    from matplotlib.ft2font import FT2Font
    for f in fm.fontManager.ttflist:
        try:
            font_file = fm.findfont(fm.FontProperties(family=f.name))
            ft = FT2Font(font_file)
            cmap = ft.get_charmap()
            gid = cmap.get(ord(_CJK_TEST_CHAR))
            if gid is not None and gid != 0:
                return f.name
        except Exception:
            continue
    return None


def _setup_style():
    global _CJK_FONT_AVAILABLE
    cjk_font = _find_cjk_font()
    if cjk_font is not None:
        chosen = cjk_font
        _CJK_FONT_AVAILABLE = True
    else:
        chosen = "DejaVu Sans"
        _CJK_FONT_AVAILABLE = False
    plt.rcParams.update({
        "font.family": chosen,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.unicode_minus": False,
    })


def _save_fig(fig, output_dir, filename):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print(f"  [Figure] Saved: {path}")
    return filename


# ── Figure 1 ──

def _make_fig1(orig_dist, opt_dist, output_dir, num_domains=22, domain_names=None):
    _setup_style()
    domain_short = _get_domain_short(num_domains, domain_names)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    m = len(orig_dist)
    x = np.arange(m)
    width = 0.35
    ax.bar(x - width / 2, orig_dist * 100, width,
           label="Original", color=COLOR_ORIG, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, opt_dist * 100, width,
           label="Optimal", color=COLOR_OPT, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Domain")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Domain Distribution: Original vs Optimal Sampling")
    ax.set_xticks(x)
    ax.set_xticklabels(domain_short, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for bars, color in [(ax.containers[0], COLOR_ORIG), (ax.containers[1], COLOR_OPT)]:
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=6, color=color)
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig1_domain_distribution.png")


# ── Figure 2 ──

def _make_fig2(domain_weights, num_domains, num_criteria, output_dir,
               domain_names=None, quality_names=None):
    _setup_style()
    domain_short = _get_domain_short(num_domains, domain_names)
    q_names = quality_names if quality_names is not None else _DEFAULT_QUALITY_NAMES
    data = np.zeros((num_domains, num_criteria))
    for m in range(num_domains):
        start = m * num_criteria
        data[m] = domain_weights[start:start + num_criteria]
    labels = domain_short

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for n in range(num_criteria):
        ax.bar(x, data[:, n], bottom=bottom, width=0.65,
               label=q_names[n], color=QUALITY_COLORS[n % len(QUALITY_COLORS)],
               edgecolor="white", linewidth=0.4)
        bottom += data[:, n]
    ax.set_xlabel("Domain")
    ax.set_ylabel("Weight")
    ax.set_title("Quality Signal Weights per Domain (α)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=8, ncol=num_criteria)
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for i in range(len(labels)):
        row = data[i]
        max_idx = np.argmax(row)
        max_val = row[max_idx]
        if max_val > 0.20:
            prefix = sum(row[:max_idx])
            ax.text(i, prefix + max_val / 2, _DEFAULT_QUALITY_SHORT[max_idx % len(_DEFAULT_QUALITY_SHORT)],
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color="white")
    plt.tight_layout()
    return _save_fig(fig, output_dir, "fig2_quality_weights.png")


# ── Table ──

def _experiment_table(exp_outputs_dir, data_path, num_domains=22, top_k=50,
                       domain_labels_override=None, domain_names=None,
                       domain_col="domain"):
    """Generate experiment table. If domain_labels_override is provided, use it
    instead of loading from data_path (supports sharded mode)."""
    if domain_labels_override is not None:
        domain_labels = domain_labels_override
    elif os.path.isdir(data_path):
        return "*(experiment table: sharded mode, see pipeline_summary.json)*"
    else:
        df = pd.read_parquet(data_path)
        domain_labels = df[domain_col].to_numpy(dtype=np.int64)
    exp_dirs = sorted([
        d for d in os.listdir(exp_outputs_dir)
        if d.startswith("exp_") and os.path.isdir(os.path.join(exp_outputs_dir, d))
    ])
    if not exp_dirs:
        return "*(no experiment data)*"

    rows = []
    domain_short = _get_domain_short(num_domains, domain_names)
    for exp_dir_name in exp_dirs[:top_k]:
        exp_dir = os.path.join(exp_outputs_dir, exp_dir_name)
        meta_path = os.path.join(exp_dir, "meta.json")
        indices_path = os.path.join(exp_dir, "selected_indices.npy")
        if not os.path.exists(indices_path):
            continue
        selected_idx = np.load(indices_path)
        sel_domain = domain_labels[selected_idx]
        dist = np.bincount(sel_domain[sel_domain >= 0],
                           minlength=num_domains).astype(np.float64)
        dist /= max(1, dist.sum())
        top5 = sorted(range(num_domains), key=lambda m: dist[m], reverse=True)[:5]
        top5_cells = " ".join(
            f"**{domain_short[m]}** {dist[m]*100:.0f}%"
            for m in top5 if dist[m] > 0.01
        )
        val_loss = "?"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            val_loss = f"{meta.get('val_loss', '?'):.4f}"
        rows.append((exp_dir_name.replace("exp_", ""), val_loss, len(selected_idx), top5_cells))

    if not rows:
        return "*(no usable experiment data)*"

    lines = [
        "### Table 1: 各实验的 Top-5 域分布\n",
        "| Exp | Val Loss | Docs | Top-5 Domain Distribution |",
        "|:---:|:--------:|:----:|:--------------------------|",
    ]
    for e, vl, docs, t5 in rows:
        lines.append(f"| {e:>4s} | {vl} | {docs:,} | {t5} |")
    losses = []
    for _, vl, _, _ in rows:
        try:
            losses.append(float(vl))
        except ValueError:
            pass
    if losses:
        lines.append(
            f"| **Avg** | **{np.mean(losses):.4f}** | **{np.mean([row[2] for row in rows]):.0f}** | "
            f"min: **{np.min(losses):.4f}**, max: **{np.max(losses):.4f}** |"
        )
    lines.append("")
    return "\n".join(lines)


# ── Main ──

def generate_report(
    output_dir, data_path, optimal_params, optimal_selected_indices,
    domain_labels, token_counts, num_domains=22, num_criteria=5,
    config=None, metrics=None, elapsed=None,
    use_sharded=False, reliability=None, proxy_loss_stats=None,
    per_task_analysis=None, dataset_size_prediction=None, stage_times=None,
    domain_names=None, quality_names=None, domain_col="domain",
):
    """Generate MD report with separate PNG figures."""
    # Compute distributions
    orig_dist = np.bincount(domain_labels[domain_labels >= 0], minlength=num_domains).astype(np.float64)
    orig_dist /= max(1, orig_dist.sum())
    opt_domain = domain_labels[optimal_selected_indices]
    opt_dist = np.bincount(opt_domain[opt_domain >= 0], minlength=num_domains).astype(np.float64)
    opt_dist /= max(1, opt_dist.sum())
    domain_w = optimal_params.merge_config.domain_weights

    # Save figures
    fig1_file = _make_fig1(orig_dist, opt_dist, output_dir, num_domains, domain_names)
    fig2_file = _make_fig2(domain_w, num_domains, num_criteria, output_dir,
                           domain_names, quality_names)

    sel_tokens = token_counts[optimal_selected_indices].sum()

    # Build markdown
    parts = ["# QuaDMix 最优采样报告\n"]

    if config or metrics:
        parts.append("## 配置与指标\n")
        parts.append("| Parameter | Value |\n|:----------|:------|")
        if config:
            for k, v in config.items():
                parts.append(f"| {k} | {v} |")
        if metrics:
            all_none = all(v is None for v in metrics.values())
            for k, v in metrics.items():
                if v is None:
                    parts.append(f"| {k} | — |")
                elif isinstance(v, float):
                    parts.append(f"| {k} | {v:.4f} |")
                else:
                    parts.append(f"| {k} | {v} |")
            if all_none:
                parts.append("")
                parts.append("> Per-task metrics are N/A: validation set has no task labels (aggregate-only mode).")
        if elapsed is not None:
            parts.append(f"| Duration | {elapsed:.1f}s |")
        parts.append("")

    if reliability:
        parts.append("## Model Reliability\n")
        parts.append("**Aggregate model** (single LightGBM on total loss, diagnostic only — not used for search):\n")
        bootstrap = reliability.get("bootstrap") or {}
        bootstrap_mean = bootstrap.get("mean")
        ci_lower = bootstrap.get("ci_lower")
        ci_upper = bootstrap.get("ci_upper")
        ci_std = bootstrap.get("ci_std")
        n_ensemble = bootstrap.get("n_ensemble_models")
        ci_width = (ci_upper - ci_lower) if (ci_lower is not None and ci_upper is not None) else None
        sample_sufficient = reliability.get("sample_sufficient")
        overfit_gap = reliability.get("overfit_gap")
        n_features = reliability.get("n_features")
        n_train = reliability.get("n_train_samples")

        parts.append("| Metric | Value | Status |")
        parts.append("|:-------|:------|:-------|")

        if bootstrap_mean is not None:
            quality = "✓ Excellent" if bootstrap_mean > 0.8 else ("✓ Good" if bootstrap_mean > 0.6 else "⚠️ Weak signal")
            parts.append(f"| **Val R² (bootstrap mean)** | **{bootstrap_mean:.4f}** | **{quality}** |")

        agg_val_r2 = metrics.get("aggregate_val_r2", metrics.get("val_r2")) if metrics else None
        if agg_val_r2 is not None:
            parts.append(f"| Val R² (single split) | {agg_val_r2:.4f} | — |")

        if ci_lower is not None and ci_upper is not None:
            ci_status = "✓ Stable" if ci_width is not None and ci_width < 0.3 else "⚠️ Wide CI"
            parts.append(f"| 95% CI | [{ci_lower:.3f}, {ci_upper:.3f}] | width={ci_width:.3f} {ci_status} |")

        if ci_std is not None:
            parts.append(f"| CI Std | {ci_std:.4f} | — |")

        if n_ensemble is not None:
            parts.append(f"| Ensemble Models | {n_ensemble} | — |")

        if overfit_gap is not None:
            gap_status = "✓ OK" if overfit_gap < 0.3 else "⚠️ Overfitting"
            parts.append(f"| Train-Val Gap | {overfit_gap:.3f} | {gap_status} |")

        if n_train is not None and n_features is not None:
            ratio = n_train / n_features if n_features > 0 else 0
            suff_status = "✓ OK" if sample_sufficient else "⚠️ Underdetermined"
            parts.append(f"| Samples/Features | {n_train}/{n_features} ({ratio:.1f}x) | {suff_status} |")

        parts.append("")

        warnings = []
        if ci_width is not None and ci_width > 0.5:
            warnings.append(f"- ⚠️ **CI width = {ci_width:.3f}**: results unreliable, increase experiments to 100+")
        if not sample_sufficient:
            warnings.append(f"- ⚠️ **Samples ({n_train}) < Features ({n_features})**: model is underdetermined, increase experiments to {n_features * 3 if n_features else 200}+")
        if overfit_gap is not None and overfit_gap > 0.3:
            warnings.append(f"- ⚠️ **Train-Val gap = {overfit_gap:.3f}**: possible overfitting, consider more experiments")
        if bootstrap_mean is not None and bootstrap_mean < 0.6:
            warnings.append(f"- ⚠️ **Bootstrap mean R² = {bootstrap_mean:.3f}**: signal too weak, consider larger model or more training steps")

        if warnings:
            parts.append("**Recommendations:**\n")
            parts.extend(warnings)
            parts.append("")

    if metrics:
        ens_r2 = metrics.get("ensemble_val_r2")
        ens_mae = metrics.get("ensemble_val_mae")
        eq_r2 = metrics.get("equal_weight_r2")
        eq_mae = metrics.get("equal_weight_mae")
        sp = metrics.get("spearman_corr")
        tk = metrics.get("top_k_recall")
        tk_val = metrics.get("top_k_value")
        sl = metrics.get("search_lift")
        if ens_r2 is not None:
            parts.append("## Model Evaluation Metrics\n")
            _swm = config.get("search_weight_mode", "equal_weight") if config else "equal_weight"
            if _swm == "equal_weight":
                mode_label = "equal-weight"
            elif _swm == "r2_sigma_weighted":
                mode_label = "R²×σ-weighted"
            else:
                mode_label = "R²-weighted"
            search_desc = "Σ z_predᵢ / K" if mode_label == "equal-weight" else "Σ wᵢ·z_predᵢ"
            parts.append(f"**Search mode:** {mode_label} (optimizes {search_desc}, matches downstream goal)\n")
            parts.append("")
            parts.append("Four complementary metrics assess prediction and search quality:\n")
            parts.append("")
            parts.append("| Metric | Value | Formula | Purpose |")
            parts.append("|:-------|:------|:--------|:--------|")
            quality = "✓ Excellent" if ens_r2 > 0.6 else ("✓ Good" if ens_r2 > 0.3 else "⚠️ Weak")
            parts.append(f"| **Overall Val R²** | **{ens_r2:.4f}** ({quality}) | R²(Σ wᵢ·z_predᵢ, Σ wᵢ·z_actualᵢ) | Search objective quality |")
            parts.append(f"| Overall Val MAE | {ens_mae:.4f} | z-score space | — |")
            if eq_r2 is not None:
                eq_quality = "✓ Excellent" if eq_r2 > 0.6 else ("✓ Good" if eq_r2 > 0.3 else "⚠️ Weak")
                parts.append(f"| **Equal-Wt Val R²** | **{eq_r2:.4f}** ({eq_quality}) | R²((1/K)Σ z_predᵢ, (1/K)Σ z_actualᵢ) | Downstream goal quality |")
                parts.append(f"| Equal-Wt Val MAE | {eq_mae:.4f} | z-score space | — |")
            if sp is not None:
                sp_quality = "✓ Excellent" if sp > 0.7 else ("✓ Good" if sp > 0.5 else ("⚠️ Moderate" if sp > 0.3 else "⚠️ Weak"))
                parts.append(f"| **Spearman Rank Corr** | **{sp:.4f}** ({sp_quality}) | corr(rank(pred), rank(actual)) | Ranking ({mode_label}, auto-adapted) |")
            if tk is not None:
                tk_quality = "✓ Excellent" if tk > 0.7 else ("✓ Good" if tk > 0.5 else ("⚠️ Moderate" if tk > 0.3 else "⚠️ Weak"))
                parts.append(f"| **Top-{tk_val} Recall** | **{tk:.4f}** ({tk_quality}) | |pred_top_k ∩ actual_top_k| / k | Hit rate ({mode_label}, auto-adapted) |")
            if sl is not None:
                sl_quality = "✓ Excellent" if sl > 1.0 else ("✓ Good" if sl > 0.5 else ("⚠️ Moderate" if sl > 0.2 else "⚠️ Weak"))
                parts.append(f"| **Search Lift** | **{sl:.4f}** σ ({sl_quality}) | (μ_random - μ_search_top_k) / σ | vs random ({mode_label}, auto-adapted) |")
            parts.append("")
            parts.append("### Interpretation\n")
            parts.append("")
            parts.append("**Overall Val R²** (search objective):\n")
            parts.append("- Uses R²-weighted z-score combination: high-R² tasks contribute more\n")
            parts.append("- Reduces noise impact from low-signal tasks\n")
            parts.append("- Directly measures how well the search strategy's predictions match reality\n")
            parts.append("- High value → search will find good parameters\n")
            parts.append("")
            parts.append("**Equal-Wt Val R²** (downstream goal):\n")
            parts.append("- Uses equal-weight average in z-score space\n")
            parts.append("- Matches downstream evaluation (21 benchmarks equally weighted, normalized)\n")
            parts.append("- Diagnostic: shows prediction quality when all tasks treated equally\n")
            parts.append("")
            parts.append(f"**Spearman Rank Correlation** (ranking ability, {mode_label}):\n")
            parts.append("- Measures whether the model correctly ranks which parameters are better\n")
            parts.append(f"- Auto-adapts: both pred and actual use {mode_label} aggregation\n")
            parts.append("- Search only needs correct ranking, not accurate absolute values\n")
            parts.append("- R² can be low when loss variance is small, but ranking may still be effective\n")
            parts.append("- High value → model knows which parameters are better\n")
            parts.append("")
            parts.append(f"**Top-K Recall** (search hit rate, {mode_label}):\n")
            parts.append("- Fraction of search's top-K predictions that are actually in the top-K\n")
            parts.append(f"- Auto-adapts: both pred and actual use {mode_label} aggregation\n")
            parts.append("- Directly measures whether the selected parameters are good\n")
            parts.append("- Most practical metric: answers 'are my chosen parameters actually good?'\n")
            parts.append("")
            parts.append(f"**Search Lift** (search value vs random, {mode_label}):\n")
            parts.append("- How many standard deviations better search's top-K is compared to random selection\n")
            parts.append(f"- Auto-adapts: uses {mode_label} actual values for evaluation\n")
            parts.append("- Positive value means search finds better parameters than random\n")
            parts.append("- Most intuitive metric for stakeholders: answers 'how much better is search than random?'\n")
            parts.append("")
            if sp is not None and sp > 0.5:
                parts.append("✓ **Spearman > 0.5**: Ranking is reliable, search results trustworthy\n")
            elif sp is not None and sp > 0.3:
                parts.append("⚠️ **Spearman 0.3-0.5**: Moderate ranking ability, search may find decent parameters\n")
            elif sp is not None:
                parts.append("⚠️ **Spearman < 0.3**: Weak ranking ability, search results unreliable\n")
            
            if tk is not None and tk > 0.5:
                parts.append(f"✓ **Top-{tk_val} Recall > 0.5**: Search finds good parameters\n")
            elif tk is not None and tk > 0.3:
                parts.append(f"⚠️ **Top-{tk_val} Recall 0.3-0.5**: Search finds some good parameters\n")
            elif tk is not None:
                parts.append(f"⚠️ **Top-{tk_val} Recall < 0.3**: Search struggles to find good parameters\n")
            
            if sl is not None and sl > 1.0:
                parts.append(f"✓ **Search Lift > 1.0σ**: Search significantly outperforms random selection\n")
            elif sl is not None and sl > 0.5:
                parts.append(f"✓ **Search Lift > 0.5σ**: Search moderately outperforms random selection\n")
            elif sl is not None and sl > 0.2:
                parts.append(f"⚠️ **Search Lift 0.2-0.5σ**: Search slightly outperforms random\n")
            elif sl is not None:
                parts.append(f"⚠️ **Search Lift < 0.2σ**: Search provides minimal advantage over random\n")
            
            if ens_r2 > 0.3 and eq_r2 is not None and eq_r2 > 0.3:
                parts.append("✓ **Both R² metrics strong**: R²-weighting and equal-weight both effective in z-score space\n")
            elif ens_r2 > 0.3 and eq_r2 is not None and eq_r2 < 0.3:
                parts.append("⚠️ **Overall strong but Equal-Wt weak**: R²-weighting outperforms equal-weight, high-R² tasks carry more signal\n")
            elif ens_r2 < 0.3 and eq_r2 is not None and eq_r2 > 0.3:
                parts.append("⚠️ **Equal-Wt strong but Overall weak**: Equal-weight outperforms R²-weighting, consider using --search-mode equal_weight\n")
            else:
                parts.append("⚠️ **Both R² metrics weak**: But check Spearman/Top-K — ranking may still be effective\n")
            parts.append("")

    if proxy_loss_stats:
        parts.append("## Proxy Experiment Loss Stats\n")
        parts.append("| Metric | Mean | Std | Min | Max |")
        parts.append("|:-------|:-----|:----|:----|:----|")
        for name, stats in proxy_loss_stats.items():
            if isinstance(stats, dict) and "mean" in stats:
                parts.append(f"| {name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
            elif isinstance(stats, float):
                parts.append(f"| {name} | {stats:.4f} | — | — | — |")
        parts.append("")

    if per_task_analysis:
        parts.append("## Per-Task Analysis (R²-Adaptive Weighting)\n")
        n_active = per_task_analysis.get("n_active", 0)
        n_filtered = per_task_analysis.get("n_filtered", 0)
        r2_method = per_task_analysis.get("r2_method", "unknown")
        search_mode = per_task_analysis.get("search_weight_mode", config.get("search_weight_mode", "") if config else "")
        parts.append(f"**Active tasks:** {n_active} | **Filtered (R²≤0):** {n_filtered} | **R² method:** {r2_method} | **Search mode:** {search_mode}\n")
        has_train_r2 = any(t.get("train_r2") is not None for t in per_task_analysis.get("tasks", []))
        if has_train_r2:
            parts.append("| Task | Train R² | Val R² | Gap | Weight | Std | Status |")
            parts.append("|:-----|--------:|-------:|----:|-------:|----:|:-------|")
        else:
            parts.append("| Task | R² | Weight | Std | Status |")
            parts.append("|:-----|---:|-------:|----:|:-------|")
        for task in per_task_analysis.get("tasks", []):
            name = task["name"]
            r2 = task["r2"]
            weight = task["weight"]
            std = task.get("std")
            std_str = f"{std:.4f}" if std is not None else "—"
            if weight == 0:
                status = "⚠️ Filtered"
            elif r2 > 0.6:
                status = "✓ Excellent"
            elif r2 > 0.3:
                status = "✓ Good"
            else:
                status = "⚠️ Weak"
            if has_train_r2:
                train_r2 = task.get("train_r2")
                gap = task.get("gap")
                train_str = f"{train_r2:.4f}" if train_r2 is not None else "—"
                gap_str = f"{gap:.4f}" if gap is not None else "—"
                parts.append(f"| {name} | {train_str} | {r2:.4f} | {gap_str} | {weight:.4f} | {std_str} | {status} |")
            else:
                parts.append(f"| {name} | {r2:.4f} | {weight:.4f} | {std_str} | {status} |")
        
        tasks = per_task_analysis.get("tasks", [])
        all_r2s = [t["r2"] for t in tasks]
        if all_r2s:
            mean_r2 = sum(all_r2s) / len(all_r2s)
            n_good = sum(1 for r in all_r2s if r > 0.3)
            parts.append("")
            parts.append(f"**Per-task R²:** mean {mean_r2:.4f} ({len(all_r2s)} tasks, {n_good} with R² > 0.3)")
            if has_train_r2:
                all_gaps = [t["gap"] for t in tasks if t.get("gap") is not None]
                if all_gaps:
                    mean_gap = sum(all_gaps) / len(all_gaps)
                    n_overfit = sum(1 for g in all_gaps if g > 0.3)
                    parts.append(f"**Overfit gap:** mean {mean_gap:.4f} ({n_overfit} tasks with gap > 0.3)")
            parts.append("")
            parts.append("Note: Per-task R² measures individual task prediction quality. Aggregate R² is lower due to correlated errors when averaging (mathematical property, not model failure).")
        parts.append("")
    else:
        parts.append("## Per-Task Analysis\n")
        parts.append("> Not available: validation set has no task labels. Search uses aggregate model only.\n")

    if dataset_size_prediction:
        parts.append("## Dataset Size Prediction\n")
        dsp = dataset_size_prediction
        parts.append("| Parameter | Value |")
        parts.append("|:----------|:------|")
        if dsp.get("total_tokens_est_B") is not None:
            parts.append(f"| Dataset Total Size | {dsp['total_tokens_est_B']:.1f}B tokens |")
        if dsp.get("omega_min") is not None:
            parts.append(f"| ω Range | [{dsp['omega_min']:.6f}, {dsp['omega_max']:.6f}] (avg {dsp['omega_avg']:.6f}) |")
        if dsp.get("estimated_tokens_B") is not None:
            parts.append(f"| Estimated Output | {dsp['estimated_tokens_B']:.2f}B tokens |")
        if dsp.get("target_tokens") is not None:
            parts.append(f"| Target | {dsp['target_tokens']/1e9:.1f}B tokens |")
        if dsp.get("note"):
            parts.append(f"| Note | {dsp['note']} |")
        parts.append("")

    if stage_times:
        parts.append("## Stage Timing\n")
        parts.append("| Stage | Time (s) | Percentage |")
        parts.append("|:------|--------:|----------:|")
        total_time = sum(stage_times.values()) or 1
        for name, secs in sorted(stage_times.items(), key=lambda x: -x[1]):
            pct = secs / total_time * 100
            parts.append(f"| {name} | {secs:.1f} | {pct:.1f}% |")
        parts.append(f"| **Total** | **{total_time:.1f}** | **100%** |")
        parts.append("")

    parts += [
        "## 采样概览\n",
        f"- **原始文档数:** {len(domain_labels):,}",
        f"- **原始总 tokens:** {int(token_counts.sum()):,}",
        f"- **最优采样文档数:** {len(optimal_selected_indices):,}",
        f"- **最优采样总 tokens:** {int(sel_tokens):,}",
        f"- **采样比例 (docs):** {len(optimal_selected_indices) / max(1, len(domain_labels)):.4f}x",
        f"- **采样比例 (tokens):** {int(sel_tokens) / max(1, int(token_counts.sum())):.4f}x",
        "",
        "---\n",
        "## Figure 1: 域分布对比\n",
        "原始数据集 vs 最优参数采样子集的域分布对比。\n",
        f"![]({fig1_file})\n",
        "---\n",
        "## Figure 2: 质量信号权重\n",
        "最优参数下各域的质量信号融合权重 α。\n",
        f"![]({fig2_file})\n",
    ]

    proxy_dir = os.path.join(output_dir, "proxy_experiments")
    if os.path.isdir(proxy_dir):
        parts += ["---\n", _experiment_table(
            proxy_dir, data_path, num_domains,
            domain_labels_override=domain_labels if use_sharded else None,
            domain_names=domain_names, domain_col=domain_col,
        )]

    parts += ["---\n", f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"]
    return "\n".join(parts)


def save_report(report, output_dir, filename="quadmix_report.md"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        f.write(report)
    print(f"[Report] Saved to: {path}  ({len(report):,} chars)")
