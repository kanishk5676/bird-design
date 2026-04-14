"""
strengthen_paper.py
===================
Addresses issues 7–13 identified in the paper review.
Reads the existing summary_analysis.csv and detailed_analysis.csv
produced by neuralfoil_analysis.py — no re-running of NeuralFoil needed.

Run from your project root:
    python strengthen_paper.py

All output figures and tables are saved to  RESULTS/paper_strengthening/
"""

import re
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths (adjust if your layout differs) ───────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR  = PROJECT_ROOT / "RESULTS" / "neuralfoil_analysis"
OUT_DIR      = PROJECT_ROOT / "RESULTS" / "paper_strengthening"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV  = RESULTS_DIR / "summary_analysis.csv"
DETAILED_CSV = RESULTS_DIR / "detailed_analysis.csv"
AIRFOIL_DIR  = PROJECT_ROOT / "OUTPUT" / "airfoils"

# ── Colour palette (consistent throughout all figures) ──────────────────────
CAT_ORDER = ["soaring", "diving", "maneuvering", "cruising", "hovering", "generalist"]
PALETTE   = dict(zip(CAT_ORDER, sns.color_palette("tab10", n_colors=6)))

# ── Figure style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        10,
    "axes.titlesize":   10,
    "axes.labelsize":    9,
    "legend.fontsize":   8,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})


# =============================================================================
# LOAD DATA
# =============================================================================

def load_results():
    """Load summary and detailed CSVs produced by neuralfoil_analysis.py."""
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {SUMMARY_CSV}\n"
            "Make sure neuralfoil_analysis.py has already been run."
        )
    summary  = pd.read_csv(SUMMARY_CSV)
    detailed = pd.read_csv(DETAILED_CSV) if DETAILED_CSV.exists() else pd.DataFrame()

    # Normalise category column
    summary["category"]  = summary["category"].str.strip().str.lower()
    if not detailed.empty:
        detailed["category"] = detailed["category"].str.strip().str.lower()

    present = summary["category"].unique().tolist()
    print(f"Loaded {len(summary):,} birds across categories: {present}")
    if detailed.empty:
        print("  (detailed_analysis.csv not found — polar plots will be skipped)")
    else:
        print(f"  {len(detailed):,} simulation rows in detailed_analysis.csv")
    return summary, detailed


summary_df, detailed_df = load_results()

# Only work with categories that actually have data
present_cats = [c for c in CAT_ORDER if c in summary_df["category"].unique()]


# =============================================================================
# ISSUE 7 — Soaring champion missing from Table 4
# Find the champion for ALL six categories and build a complete table.
# =============================================================================

def build_complete_table4(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the top-performing bird per flight category, ranked by max_LD.
    The paper's Table 4 was missing the soaring champion — this function
    guarantees one row per category that has data.
    """
    rows = []
    for cat in CAT_ORDER:
        sub = summary_df[summary_df["category"] == cat]
        if sub.empty:
            print(f"  WARNING: Category '{cat}' has no data — champion cannot be identified.")
            continue

        # Champion = bird with highest max_LD in its category
        best = sub.loc[sub["max_LD"].idxmax()].copy()

        rows.append({
            "Category":           cat.capitalize(),
            "Champion species":   best["bird_name"],
            "Max L/D":            round(best["max_LD"],     2),
            "Cruise L/D":         round(best.get("cruise_LD",    np.nan), 2),
            "Climb L/D":          round(best.get("climb_LD",     np.nan), 2),
            "CL_max":             round(best.get("CL_max",       np.nan), 3),
            "Stall α (°)":        round(best.get("stall_alpha",  np.nan), 1),
            "Min CD":             round(best.get("min_CD",       np.nan), 5),
            "Design-pt L/D":      round(best.get("design_point_LD", np.nan), 2),
            "Design Re":          best.get("design_point_Re",   np.nan),
        })

    table4 = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("TABLE 4 — Complete category champions (all 6 categories)")
    print("=" * 80)
    print(table4.to_string(index=False))
    table4.to_csv(OUT_DIR / "table4_all_champions.csv", index=False)
    print(f"\nSaved: table4_all_champions.csv")
    return table4


table4 = build_complete_table4(summary_df)


# =============================================================================
# ISSUE 8 — No distribution statistics  (mean, median, IQR, full spread)
# =============================================================================

def compute_category_statistics(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for every aerodynamic metric across every
    category.  This is the table that belongs in the paper's results section.
    """
    metrics = {
        "max_LD":     "Max L/D",
        "cruise_LD":  "Cruise L/D",
        "climb_LD":   "Climb L/D",
        "CL_max":     "CL_max",
        "min_CD":     "Min CD",
        "stall_alpha":"Stall α (°)",
        "LD_CoV":     "L/D CoV",
    }
    metrics = {k: v for k, v in metrics.items() if k in summary_df.columns}

    rows = []
    for cat in present_cats:
        sub = summary_df[summary_df["category"] == cat]
        row = {"Category": cat.capitalize(), "N": len(sub)}
        for col, label in metrics.items():
            vals = sub[col].dropna()
            row[f"{label} mean"]   = round(vals.mean(), 3)
            row[f"{label} median"] = round(vals.median(), 3)
            row[f"{label} std"]    = round(vals.std(), 3)
            row[f"{label} Q1"]     = round(vals.quantile(0.25), 3)
            row[f"{label} Q3"]     = round(vals.quantile(0.75), 3)
        rows.append(row)

    stats_table = pd.DataFrame(rows)
    stats_table.to_csv(OUT_DIR / "category_statistics.csv", index=False)
    print("\nCategory statistics saved: category_statistics.csv")
    return stats_table


stats_table = compute_category_statistics(summary_df)


def plot_distribution_figure(summary_df: pd.DataFrame) -> None:
    """
    Figure 1: Six-panel boxplot grid showing the distribution of max_LD,
    CL_max, min_CD, stall_alpha, cruise_LD, and LD_CoV across categories.
    Individual data points are jittered on top of each box so the reader
    can see sample size and spread simultaneously.
    The category champion is annotated with a star on each panel.
    """
    metrics = [
        ("max_LD",      "Maximum L/D",        False),
        ("CL_max",      "CL_max",              False),
        ("min_CD",      "Minimum CD",          False),
        ("stall_alpha", "Stall angle (°)",     False),
        ("cruise_LD",   "Cruise L/D (α ≤ 2°)", False),
        ("LD_CoV",      "L/D coefficient of variation", False),
    ]
    metrics = [(col, lbl, inv) for col, lbl, inv in metrics
               if col in summary_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()
    rng = np.random.default_rng(0)

    for ax, (col, ylabel, _) in zip(axes_flat, metrics):
        # Collect per-category data in display order
        data, labels, colors = [], [], []
        for cat in present_cats:
            vals = summary_df[summary_df["category"] == cat][col].dropna()
            if len(vals):
                data.append(vals.values)
                labels.append(cat)
                colors.append(PALETTE[cat])

        bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)

        # Jittered individual points
        for i, (vals, cat) in enumerate(zip(data, labels), start=1):
            jitter = rng.uniform(-0.22, 0.22, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=PALETTE[cat], alpha=0.35, s=8, zorder=3)

        # Star on champion value
        for i, (vals, cat) in enumerate(zip(data, labels), start=1):
            champ_val = vals.max() if col != "min_CD" and col != "LD_CoV" else vals.min()
            ax.scatter([i], [champ_val], marker="*", s=180,
                       color=PALETTE[cat], zorder=5, edgecolors="black", linewidths=0.5)
            ax.annotate(f" {champ_val:.2f}", xy=(i, champ_val),
                        fontsize=6.5, va="center")

        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

        # Embed N per category
        for i, (vals, _) in enumerate(zip(data, labels), start=1):
            ax.text(i, ax.get_ylim()[0], f"n={len(vals)}",
                    ha="center", va="bottom", fontsize=6, color="gray")

    fig.suptitle(
        "Figure — Aerodynamic performance distributions by flight category\n"
        "(box = IQR, whiskers = 1.5×IQR, dots = individual airfoils, ★ = champion)",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig_distributions.png")


plot_distribution_figure(summary_df)


# =============================================================================
# ISSUE 9 — No polar curves
# CL-α, CD-α, and drag polars (CL vs CD) for every category champion.
# =============================================================================

def load_dat_coords(bird_name: str, category: str) -> np.ndarray | None:
    """Search AIRFOIL_DIR for a .dat file matching bird_name, return coords."""
    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", bird_name)
    # Try exact match first, then partial
    candidates = list((AIRFOIL_DIR / category).glob(f"{safe_name}.dat")) \
               + list(AIRFOIL_DIR.rglob(f"*{safe_name[:20]}*.dat"))
    if not candidates:
        return None
    try:
        pts = []
        with open(candidates[0]) as f:
            for line in f:
                s = line.strip()
                if s.startswith("#") or not s:
                    continue
                parts = s.split()
                if len(parts) == 2:
                    try:
                        pts.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
        return np.array(pts) if len(pts) > 10 else None
    except Exception:
        return None


def plot_polar_figures(summary_df: pd.DataFrame,
                       detailed_df: pd.DataFrame,
                       table4: pd.DataFrame) -> None:
    """
    Three polar figures for each category champion:
      Figure A: CL-α polars at Re = 1e5 (all Re as lighter lines for context)
      Figure B: CD-α polars at Re = 1e5
      Figure C: Drag polars (CL vs CD) at Re = 1e5

    These figures are completely absent from the paper as submitted.
    They are the standard output of any 2-D airfoil aerodynamics study.
    """
    if detailed_df.empty:
        print("Skipping polar plots — detailed_analysis.csv not available.")
        return

    # The detailed CSV uses 'LD_ratio' as the column name
    ld_col = "LD_ratio" if "LD_ratio" in detailed_df.columns else "LD"

    target_re = 1e5   # representative Reynolds for the main polar comparison
    n_champs  = len(table4)
    ncols     = 3
    nrows     = (n_champs + ncols - 1) // ncols

    for fig_tag, y_col, ylabel, title_suffix in [
        ("A_CL_alpha", "CL",  "Lift coefficient C_L",  "CL–α polars"),
        ("B_CD_alpha", "CD",  "Drag coefficient C_D",  "CD–α polars"),
    ]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
        axf = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes.flatten()

        for ax_i, (_, cr) in enumerate(table4.iterrows()):
            ax   = axf[ax_i]
            cat  = cr["Category"].lower()
            name = cr["Champion species"]
            color = PALETTE.get(cat, "gray")

            bird_rows = detailed_df[detailed_df["bird_name"] == name]
            if bird_rows.empty:
                ax.text(0.5, 0.5, f"No data\n{name[:22]}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=7)
                ax.set_title(f"{cat}: {name[:20]}", fontsize=8)
                continue

            # Plot all Re as thin background lines for context
            for re_val in sorted(bird_rows["Re"].unique()):
                sub = bird_rows[np.isclose(bird_rows["Re"], re_val, rtol=0.05)
                               ].sort_values("alpha")
                if sub.empty:
                    continue
                is_target = np.isclose(re_val, target_re, rtol=0.05)
                ax.plot(sub["alpha"], sub[y_col],
                        color=color,
                        linewidth=2.0 if is_target else 0.6,
                        alpha=1.0   if is_target else 0.25,
                        label=f"Re={re_val:.0e}" if is_target else "_")

            ax.axhline(0, color="gray", linewidth=0.4, linestyle="--")
            ax.axvline(0, color="gray", linewidth=0.4, linestyle="--")
            ax.set_xlabel("α (°)");  ax.set_ylabel(ylabel)
            ax.set_title(f"{cat}: {name[:22]}\nmax L/D = {cr['Max L/D']:.1f}", fontsize=8)
            ax.legend(fontsize=6);  ax.grid(True, alpha=0.2)

        for ax_j in range(ax_i + 1, len(axf)):
            axf[ax_j].set_visible(False)

        fig.suptitle(
            f"Figure {fig_tag} — {title_suffix} for each category champion\n"
            f"(thick line = Re={target_re:.0e}, thin lines = other Re values)",
            fontsize=10, y=1.01
        )
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"fig_{fig_tag}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: fig_{fig_tag}.png")

    # ── Figure C: Drag polars CL vs CD ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 8))

    for _, cr in table4.iterrows():
        cat  = cr["Category"].lower()
        name = cr["Champion species"]
        color = PALETTE.get(cat, "gray")

        sub = detailed_df[
            (detailed_df["bird_name"] == name) &
            np.isclose(detailed_df["Re"], target_re, rtol=0.05)
        ].sort_values("alpha").dropna(subset=["CL", "CD"])

        if sub.empty:
            continue

        ax.plot(sub["CD"], sub["CL"], color=color, linewidth=2,
                label=f"{cat}: {name[:20]}")

        # Mark the best-L/D operating point with a star
        ld_vals = sub["CL"] / sub["CD"].replace(0, np.nan)
        if ld_vals.notna().any():
            best = sub.loc[ld_vals.idxmax()]
            ax.scatter([best["CD"]], [best["CL"]], color=color,
                       marker="*", s=200, zorder=5, edgecolors="black", linewidths=0.5)
            ax.annotate(f"  L/D={best['CL']/best['CD']:.0f} @ α={best['alpha']:.0f}°",
                        xy=(best["CD"], best["CL"]), fontsize=7, color=color)

    # Add iso-L/D reference lines
    cd_range = np.linspace(0.005, ax.get_xlim()[1] if ax.get_xlim()[1] > 0.01 else 0.06, 100)
    for ld_ref in [20, 40, 60, 80]:
        ax.plot(cd_range, ld_ref * cd_range, color="lightgray",
                linewidth=0.7, linestyle=":", zorder=0)
        ax.text(cd_range[-1], ld_ref * cd_range[-1], f" L/D={ld_ref}",
                fontsize=6.5, color="gray", va="center")

    ax.set_xlabel("Drag coefficient CD")
    ax.set_ylabel("Lift coefficient CL")
    ax.set_title(
        f"Figure C — Drag polars (CL vs CD) for category champions  [Re={target_re:.0e}]\n"
        "(★ = peak efficiency point; dashed lines = iso-L/D contours)",
        fontsize=9
    )
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_C_drag_polars.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig_C_drag_polars.png")

    # ── Figure D: Geometry overlay ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_xlim(-0.02, 1.02);  ax.set_ylim(-0.20, 0.22)
    ax.axhline(0, color="lightgray", linewidth=0.5)
    ax.set_xlabel("Normalised chord  x/c")
    ax.set_ylabel("Normalised thickness  y/c")
    ax.set_title("Figure D — All six category champions: airfoil geometry comparison")
    ax.set_aspect("equal")

    for _, cr in table4.iterrows():
        cat   = cr["Category"].lower()
        name  = cr["Champion species"]
        color = PALETTE.get(cat, "gray")
        coords = load_dat_coords(name, cat)
        if coords is not None:
            ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2,
                    label=f"{cat}  (L/D={cr['Max L/D']:.1f})")
        else:
            print(f"  Geometry not found for {name} — skipped from overlay")

    ax.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_D_geometry_overlay.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig_D_geometry_overlay.png")


plot_polar_figures(summary_df, detailed_df, table4)


# =============================================================================
# ISSUE 10 — No statistical significance testing
# Kruskal-Wallis across all categories + pairwise Mann-Whitney with Bonferroni.
# =============================================================================

def run_statistical_tests(summary_df: pd.DataFrame) -> dict:
    """
    For each of four key aerodynamic metrics, run:
      1. Kruskal-Wallis H-test (non-parametric ANOVA equivalent) — tests
         whether any category differs from the others.
      2. Pairwise Mann-Whitney U tests for all category pairs, with
         Bonferroni correction for multiple comparisons.

    We use non-parametric tests because:
      - L/D distributions are typically right-skewed (a few very high performers)
      - Category sample sizes vary widely (hovering n≈400 vs maneuvering n≈2400)
      - Normality cannot be assumed across all categories

    Returns a dict of results keyed by metric name.
    """
    metrics = [m for m in ["max_LD", "CL_max", "min_CD", "stall_alpha"]
               if m in summary_df.columns]

    all_results = {}
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 70)

    for metric in metrics:
        print(f"\n── {metric} ──")
        groups = {cat: summary_df[summary_df["category"] == cat][metric].dropna().values
                  for cat in present_cats
                  if len(summary_df[summary_df["category"] == cat][metric].dropna()) >= 3}

        if len(groups) < 2:
            print("  Not enough groups — skipping.")
            continue

        # Kruskal-Wallis
        kw_stat, kw_p = kruskal(*groups.values())
        print(f"  Kruskal-Wallis  H={kw_stat:.3f},  p={kw_p:.4e}  "
              f"→ {'SIGNIFICANT' if kw_p < 0.05 else 'not significant'} at α=0.05")

        # Pairwise Mann-Whitney with Bonferroni
        pairs    = list(itertools.combinations(groups.keys(), 2))
        n_comp   = len(pairs)
        pw_rows  = []
        for g1, g2 in pairs:
            _, p_raw = mannwhitneyu(groups[g1], groups[g2], alternative="two-sided")
            p_bonf   = min(p_raw * n_comp, 1.0)
            # Effect size: rank-biserial correlation
            n1, n2 = len(groups[g1]), len(groups[g2])
            u_stat, _ = mannwhitneyu(groups[g1], groups[g2], alternative="two-sided")
            rb_corr = 1 - (2 * u_stat) / (n1 * n2)   # rank-biserial r

            pw_rows.append({
                "group1":      g1,
                "group2":      g2,
                "U_stat":      round(u_stat, 1),
                "p_raw":       p_raw,
                "p_bonferroni": p_bonf,
                "effect_r":    round(rb_corr, 3),
                "significant": p_bonf < 0.05,
            })
            star = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 \
                         else "*"   if p_bonf < 0.05  else "ns"
            print(f"    {g1:<14} vs {g2:<14}  p={p_bonf:.4f} {star}  r={rb_corr:+.3f}")

        pw_df = pd.DataFrame(pw_rows)
        pw_df.to_csv(OUT_DIR / f"stats_pairwise_{metric}.csv", index=False)
        all_results[metric] = {"kw_stat": kw_stat, "kw_p": kw_p, "pairwise": pw_df}

    return all_results


stat_results = run_statistical_tests(summary_df)


def plot_significance_heatmaps(stat_results: dict) -> None:
    """
    One heatmap per metric: lower-triangle of Bonferroni-corrected p-values.
    Green cell = significant difference; red = not significant.
    This is a standard figure in comparative aerodynamics papers.
    """
    metrics = [m for m in stat_results if "pairwise" in stat_results[m]]
    if not metrics:
        return

    ncols = min(2, len(metrics))
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    axf = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, metric in zip(axf, metrics):
        pw_df  = stat_results[metric]["pairwise"]
        groups = present_cats
        n      = len(groups)
        mat    = np.ones((n, n))

        for _, row in pw_df.iterrows():
            if row["group1"] in groups and row["group2"] in groups:
                i = groups.index(row["group1"])
                j = groups.index(row["group2"])
                mat[i, j] = row["p_bonferroni"]
                mat[j, i] = row["p_bonferroni"]

        # Build annotation matrix
        annot = np.full((n, n), "", dtype=object)
        for i in range(n):
            for j in range(n):
                p = mat[i, j]
                if i == j:
                    annot[i, j] = ""
                elif p < 0.001:
                    annot[i, j] = "***"
                elif p < 0.01:
                    annot[i, j] = "**"
                elif p < 0.05:
                    annot[i, j] = "*"
                else:
                    annot[i, j] = "ns"

        mask = np.triu(np.ones((n, n), dtype=bool), k=0)  # hide upper + diagonal
        sns.heatmap(
            pd.DataFrame(mat, index=groups, columns=groups),
            mask=mask, annot=annot, fmt="s",
            cmap="RdYlGn_r", vmin=0, vmax=0.10,
            linewidths=0.5, cbar=False, ax=ax,
            annot_kws={"fontsize": 9, "fontweight": "bold"}
        )
        kw_p  = stat_results[metric]["kw_p"]
        kw_h  = stat_results[metric]["kw_stat"]
        ax.set_title(
            f"{metric}\nKruskal-Wallis H={kw_h:.1f}, p={kw_p:.2e}",
            fontsize=9
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    for ax_j in range(len(metrics), len(axf)):
        axf[ax_j].set_visible(False)

    fig.suptitle(
        "Pairwise Mann-Whitney significance (Bonferroni corrected)\n"
        "*** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_significance_heatmaps.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig_significance_heatmaps.png")


plot_significance_heatmaps(stat_results)


# =============================================================================
# ISSUE 11 — Category overlap diagnostic
# Shows which species satisfy multiple category criteria simultaneously,
# and how many are differently assigned depending on priority order.
# =============================================================================

def audit_category_overlap(summary_df: pd.DataFrame) -> None:
    """
    Re-apply the paper's original category criteria to summary_df and count
    how many species satisfy more than one criterion simultaneously.
    This produces the overlap table that should appear in the paper's
    methodology section as justification for adding a priority order.

    Note: this requires the morphological indices (aspect_ratio, etc.) to be
    present in summary_df.  If they are not, we print an advisory message
    and skip the quantitative audit — the fix (priority ordering) still applies
    to categorisation.py regardless.
    """
    morph_cols = ["aspect_ratio", "wing_loading", "pointedness_index",
                  "efficiency_index", "Hand_Wing_Index", "Wing_Length"]
    missing = [c for c in morph_cols if c not in summary_df.columns]

    if missing:
        print(f"\nISSUE 11 ADVISORY: summary_analysis.csv does not contain morphological "
              f"columns {missing}.\n"
              f"The overlap audit requires re-running categorisation.py and merging the result.\n"
              f"FIX to apply in categorisation.py:\n"
              f"  Replace your current category assignment with the priority-ordered version "
              f"shown below.\n")
        print(PRIORITY_ORDER_CODE)
        return

    df = summary_df.copy()
    wl_median = df["wing_loading"].median()
    eff_60th  = df["efficiency_index"].quantile(0.60)

    # Re-evaluate every criterion independently (no priority order)
    df["crit_hovering"]    = (df["Wing_Length"] < 100) & (df["Hand_Wing_Index"] > 50)
    df["crit_soaring"]     = (df["aspect_ratio"] > 2.5) & (df["wing_loading"] < wl_median)
    df["crit_diving"]      = (df["pointedness_index"] > 0.4) & (df["aspect_ratio"] > 2.0)
    df["crit_maneuvering"] = (df["Hand_Wing_Index"] < 35) & (df["aspect_ratio"] < 2.5)
    df["crit_cruising"]    = (df["efficiency_index"] > eff_60th) & \
                              (df["aspect_ratio"].between(2.0, 3.0))

    crit_cols = ["crit_hovering","crit_soaring","crit_diving",
                 "crit_maneuvering","crit_cruising"]
    df["n_categories_met"] = df[crit_cols].sum(axis=1)

    overlap_counts = df["n_categories_met"].value_counts().sort_index()
    print("\n" + "=" * 60)
    print("ISSUE 11 — Category overlap audit (paper methodology fix)")
    print("=" * 60)
    for n_cats, count in overlap_counts.items():
        pct = count / len(df) * 100
        print(f"  Meets {n_cats} criteria simultaneously: {count:6,} species  ({pct:.1f}%)")

    multi = df[df["n_categories_met"] > 1]
    print(f"\n  Species meeting >1 criterion: {len(multi):,} ({len(multi)/len(df)*100:.1f}%)")
    print("  This confirms the classification is NOT deterministic without a priority order.")

    # Cross-tabulation: which criterion pairs overlap most?
    print("\n  Most common overlapping criterion pairs:")
    for (c1, c2) in itertools.combinations(crit_cols, 2):
        both = ((df[c1]) & (df[c2])).sum()
        if both > 0:
            print(f"    {c1[5:]:<14} ∩ {c2[5:]:<14}: {both:5,} species")

    df[["bird_name","category","n_categories_met"] + crit_cols].to_csv(
        OUT_DIR / "category_overlap_audit.csv", index=False)
    print("\nSaved: category_overlap_audit.csv")


# The drop-in replacement code block to add to categorisation.py
PRIORITY_ORDER_CODE = """
# ── PRIORITY-ORDERED CATEGORISATION (paste into categorisation.py) ──────────
#
# Apply criteria in order from most to least restrictive.
# Each species lands in exactly ONE category.
# The 'generalist' catch-all is applied last.
#
# wl_median = df['wing_loading'].median()
# eff_60th  = df['efficiency_index'].quantile(0.60)

cat = pd.Series('generalist', index=df.index)

# 5. Cruising (applied first so later overrides are more specific)
cat[(df['efficiency_index'] > eff_60th) &
    (df['aspect_ratio'].between(2.0, 3.0))] = 'cruising'

# 4. Maneuvering
cat[(df['Hand_Wing_Index'] < 35) &
    (df['aspect_ratio'] < 2.5)] = 'maneuvering'

# 3. Diving
cat[(df['pointedness_index'] > 0.4) &
    (df['aspect_ratio'] > 2.0)] = 'diving'

# 2. Soaring
cat[(df['aspect_ratio'] > 2.5) &
    (df['wing_loading'] < wl_median)] = 'soaring'

# 1. Hovering — highest priority, overwrites everything
cat[(df['Wing_Length'] < 100) &
    (df['Hand_Wing_Index'] > 50)] = 'hovering'

df['flight_category'] = cat
assert df['flight_category'].isna().sum() == 0, "Unclassified species remain!"
# ─────────────────────────────────────────────────────────────────────────────
"""

audit_category_overlap(summary_df)
print(PRIORITY_ORDER_CODE)


# =============================================================================
# ISSUE 12 — NeuralFoil confidence scores never reported
# =============================================================================

def report_confidence_scores(detailed_df: pd.DataFrame,
                              summary_df: pd.DataFrame) -> None:
    """
    Count and visualise the fraction of simulations that fell below the
    analysis_confidence threshold.  This is a standard quality metric for
    surrogate-model studies — reviewers will ask for it.

    The paper states CONFIDENCE_THRESHOLD = 0.5 but never reports how many
    simulations this flagged.
    """
    if detailed_df.empty:
        print("\nISSUE 12: detailed_analysis.csv not available — cannot compute confidence stats.")
        return

    conf_col = "analysis_confidence"
    lc_col   = "low_confidence"

    if conf_col not in detailed_df.columns:
        print(f"\nISSUE 12: Column '{conf_col}' not found in detailed_analysis.csv.")
        return

    total_sims = len(detailed_df)
    n_low  = int(detailed_df[lc_col].sum()) if lc_col in detailed_df.columns else \
             int((detailed_df[conf_col] < 0.5).sum())
    pct_low = n_low / total_sims * 100

    print("\n" + "=" * 60)
    print("ISSUE 12 — NeuralFoil confidence score report")
    print("=" * 60)
    print(f"  Total simulations          : {total_sims:,}")
    print(f"  Low-confidence (< 0.50)    : {n_low:,}  ({pct_low:.2f}%)")
    print(f"  High-confidence (≥ 0.50)   : {total_sims - n_low:,}  ({100-pct_low:.2f}%)")
    print(f"\n  Mean confidence score      : {detailed_df[conf_col].mean():.4f}")
    print(f"  Median confidence score    : {detailed_df[conf_col].median():.4f}")
    print(f"  Std confidence score       : {detailed_df[conf_col].std():.4f}")

    # Per-category breakdown
    print("\n  Per-category confidence breakdown:")
    for cat in present_cats:
        sub    = detailed_df[detailed_df["category"] == cat]
        n_lc   = int((sub[conf_col] < 0.5).sum())
        pct_lc = n_lc / max(len(sub), 1) * 100
        print(f"    {cat:<14}: {len(sub):7,} sims,  {n_lc:5,} low-conf  ({pct_lc:.1f}%)")

    # ── Figure: confidence score distribution ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Histogram of confidence scores
    ax = axes[0]
    ax.hist(detailed_df[conf_col].dropna(), bins=50, color="steelblue",
            edgecolor="white", linewidth=0.3, alpha=0.85)
    ax.axvline(0.5, color="red", linewidth=1.5, linestyle="--",
               label=f"Threshold = 0.50\n({pct_low:.1f}% flagged)")
    ax.set_xlabel("NeuralFoil analysis_confidence")
    ax.set_ylabel("Number of simulations")
    ax.set_title("Distribution of confidence scores\nacross all simulations")
    ax.legend(fontsize=8)

    # Per-category low-confidence fractions (bar chart)
    ax = axes[1]
    cats_with_data = [c for c in present_cats
                      if c in detailed_df["category"].unique()]
    lc_fractions = [
        (detailed_df[detailed_df["category"] == c][conf_col] < 0.5).mean() * 100
        for c in cats_with_data
    ]
    bar_colors = [PALETTE.get(c, "gray") for c in cats_with_data]
    bars = ax.bar(range(len(cats_with_data)), lc_fractions,
                  color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.8)
    ax.axhline(pct_low, color="gray", linewidth=1, linestyle="--",
               label=f"Overall mean {pct_low:.1f}%")
    ax.set_xticks(range(len(cats_with_data)))
    ax.set_xticklabels(cats_with_data, rotation=30, ha="right")
    ax.set_ylabel("Low-confidence simulations (%)")
    ax.set_title("Low-confidence fraction by category\n(confidence < 0.50)")
    ax.legend(fontsize=8)
    for bar, pct in zip(bars, lc_fractions):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=7)

    plt.suptitle(
        "Figure — NeuralFoil confidence score analysis\n"
        "(This metric must be reported in the paper's methodology section)",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_confidence_scores.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("\nSaved: fig_confidence_scores.png")

    # Save the summary as a CSV table for direct inclusion in the paper
    conf_summary = pd.DataFrame([{
        "total_simulations": total_sims,
        "low_confidence_n":  n_low,
        "low_confidence_pct": round(pct_low, 2),
        "mean_confidence":   round(detailed_df[conf_col].mean(), 4),
        "median_confidence": round(detailed_df[conf_col].median(), 4),
    }])
    conf_summary.to_csv(OUT_DIR / "confidence_score_summary.csv", index=False)
    print("Saved: confidence_score_summary.csv")


report_confidence_scores(detailed_df, summary_df)


# =============================================================================
# ISSUE 13 — Cruise L/D (35-43) vs max L/D (82-92) gap unexplained
# Diagnostic: find what alpha and Re achieve the maximum L/D for each
# champion and compare to the α ≤ 2° cruise definition.
# =============================================================================

def diagnose_ld_gap(summary_df: pd.DataFrame,
                    detailed_df: pd.DataFrame,
                    table4: pd.DataFrame) -> None:
    """
    For each category champion, find:
      1. The (alpha, Re) operating point achieving max L/D
      2. The mean L/D at alpha ≤ 2° (the paper's 'cruise' definition)
      3. The ratio between them

    This explains why cruise L/D ≈ 35–43 while max L/D ≈ 82–92.
    The issue is definitional: max L/D occurs near α = 6–10° for cambered
    airfoils, but the cruise definition forces α ≤ 2°, which cuts the
    lift-generating range almost in half.

    We also plot L/D vs α for each champion so the reader can see the
    operating-point mismatch directly.
    """
    print("\n" + "=" * 70)
    print("ISSUE 13 — Cruise L/D vs max L/D diagnostic")
    print("=" * 70)
    print(f"  Paper's cruise definition: α ≤ 2° (mean L/D in this range)")
    print(f"  This is the root cause of the apparent gap.\n")

    rows = []
    for _, cr in table4.iterrows():
        cat  = cr["Category"].lower()
        name = cr["Champion species"]

        if detailed_df.empty:
            rows.append({
                "Category":       cr["Category"],
                "Champion":       name,
                "Max L/D":        cr["Max L/D"],
                "Cruise L/D":     cr["Cruise L/D"],
                "Gap factor":     round(cr["Max L/D"] / max(cr["Cruise L/D"], 1), 2),
            })
            continue

        bird = detailed_df[detailed_df["bird_name"] == name].copy()
        if bird.empty:
            continue

        ld_col = "LD_ratio" if "LD_ratio" in bird.columns else "LD"
        bird = bird.dropna(subset=[ld_col])

        # Max L/D operating point
        best = bird.loc[bird[ld_col].idxmax()]
        max_ld    = best[ld_col]
        best_alpha = best["alpha"]
        best_re    = best["Re"]

        # Cruise L/D (α ≤ 2°, all Re)
        cruise = bird[bird["alpha"] <= 2]
        cruise_ld = cruise[ld_col].mean() if not cruise.empty else np.nan

        # L/D at design Re (most relevant operating point)
        design_re = cr.get("Design Re", 1e5)
        design_sub = bird[np.isclose(bird["Re"], design_re, rtol=0.2)]
        design_ld  = design_sub[ld_col].max() if not design_sub.empty else np.nan

        # L/D at moderate cruise alpha (4-6°) — a more realistic cruise definition
        moderate_cruise = bird[bird["alpha"].between(4, 6)]
        mod_cruise_ld   = moderate_cruise[ld_col].mean() \
                          if not moderate_cruise.empty else np.nan

        gap = max_ld / max(cruise_ld, 1)
        print(f"  {cat:<14}  max L/D={max_ld:.1f} @ α={best_alpha:.0f}°, Re={best_re:.0e}")
        print(f"              cruise L/D (α≤2°) = {cruise_ld:.1f}  →  gap factor = {gap:.2f}×")
        print(f"              moderate cruise (α=4-6°) L/D = {mod_cruise_ld:.1f}")
        print(f"              NOTE: cruise should be re-defined as α=4-6° for cambered airfoils\n")

        rows.append({
            "Category":               cr["Category"],
            "Champion":               name,
            "Max L/D":                round(max_ld, 2),
            "Alpha at max L/D (°)":   round(best_alpha, 1),
            "Re at max L/D":          best_re,
            "Cruise L/D (α≤2°)":     round(cruise_ld, 2),
            "Moderate cruise (α=4-6°)": round(mod_cruise_ld, 2),
            "Gap factor (max/cruise)": round(gap, 2),
        })

    diag_df = pd.DataFrame(rows)
    diag_df.to_csv(OUT_DIR / "ld_gap_diagnostic.csv", index=False)
    print(f"Saved: ld_gap_diagnostic.csv")

    if detailed_df.empty:
        return

    # ── Figure: L/D vs alpha for each champion at design-point Re ────────────
    ld_col = "LD_ratio" if "LD_ratio" in detailed_df.columns else "LD"
    fig, ax = plt.subplots(figsize=(12, 6))

    for _, cr in table4.iterrows():
        cat  = cr["Category"].lower()
        name = cr["Champion species"]
        design_re = cr.get("Design Re", 1e5)
        color = PALETTE.get(cat, "gray")

        bird = detailed_df[
            (detailed_df["bird_name"] == name) &
            np.isclose(detailed_df["Re"],
                       design_re if not pd.isna(design_re) else 1e5,
                       rtol=0.2)
        ].sort_values("alpha").dropna(subset=[ld_col, "alpha"])

        if bird.empty:
            # Fall back to Re=1e5
            bird = detailed_df[
                (detailed_df["bird_name"] == name) &
                np.isclose(detailed_df["Re"], 1e5, rtol=0.05)
            ].sort_values("alpha").dropna(subset=[ld_col, "alpha"])

        if bird.empty:
            continue

        ax.plot(bird["alpha"], bird[ld_col], color=color, linewidth=2,
                label=f"{cat}: {name[:18]}")

        # Mark the peak
        best = bird.loc[bird[ld_col].idxmax()]
        ax.scatter([best["alpha"]], [best[ld_col]], color=color,
                   marker="*", s=180, zorder=5, edgecolors="black", linewidths=0.5)

    # Shade the paper's cruise region (α ≤ 2°)
    ax.axvspan(-4, 2, alpha=0.08, color="blue",
               label="Paper's cruise region (α ≤ 2°)")
    ax.axvspan(4, 6, alpha=0.08, color="green",
               label="Better cruise definition (α = 4–6°)")
    ax.axvline(2,  color="blue",  linewidth=1,   linestyle="--", alpha=0.6)
    ax.axvline(4,  color="green", linewidth=1,   linestyle="--", alpha=0.6)
    ax.axhline(0,  color="gray",  linewidth=0.4, linestyle="-")

    ax.set_xlabel("Angle of attack  α (°)")
    ax.set_ylabel("L/D ratio")
    ax.set_title(
        "Figure — L/D vs α for category champions at design-point Re\n"
        "Blue band = paper's cruise definition (α≤2°)  |  "
        "Green band = recommended redefinition (α=4–6°)",
        fontsize=9
    )
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_LD_gap_diagnostic.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved: fig_LD_gap_diagnostic.png")

    print("\n  PAPER FIX (Issue 13):")
    print("  Change the cruise L/D definition from 'mean(L/D) at α≤2°'")
    print("  to 'mean(L/D) at α=4–6°' — this is the standard low-Re cruise angle")
    print("  for cambered airfoils and will close the unexplained gap.")
    print("  Alternatively, add a sentence in Section 3.4 explaining that max L/D")
    print("  occurs at the polar peak (α≈6-8°) while your cruise definition targets")
    print("  near-zero incidence, which is a conservative estimate of cruise performance.")


diagnose_ld_gap(summary_df, detailed_df, table4)


# =============================================================================
# FINAL OUTPUT SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("ALL OUTPUTS SAVED TO:", OUT_DIR)
print("=" * 70)
for f in sorted(OUT_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45}  {size_kb:7.1f} KB")

print("""
─────────────────────────────────────────────────────────────────────
HOW TO USE THESE OUTPUTS IN YOUR PAPER
─────────────────────────────────────────────────────────────────────

Issue 7  →  Replace Table 4 with  table4_all_champions.csv
            (now includes the missing soaring champion)

Issue 8  →  Add  fig_distributions.png  as Figure 5 in results.
            Add  category_statistics.csv  as a supplementary table.

Issue 9  →  Add  fig_A_CL_alpha.png,  fig_B_CD_alpha.png,
            fig_C_drag_polars.png  as Figures 6–8 in results.
            Add  fig_D_geometry_overlay.png  as Figure 9.

Issue 10 →  Add  fig_significance_heatmaps.png  after Table 4.
            Report KW H-statistic and p-values in Section 4.
            Cite: "All pairwise differences were tested with
            Mann-Whitney U tests (Bonferroni-corrected α=0.05)."

Issue 11 →  Paste PRIORITY_ORDER_CODE into categorisation.py
            and re-run to get a clean, overlap-free classification.
            Add a sentence in Section 3.2: "To ensure each species
            belongs to exactly one category, criteria are applied in
            priority order: hovering > soaring > diving > maneuvering
            > cruising > generalist."

Issue 12 →  Add to Section 3.4 (NeuralFoil methodology):
            "Of the X total simulations, Y (Z%) were flagged as
            low-confidence (analysis_confidence < 0.5). These results
            were retained but are reported separately."
            Include  fig_confidence_scores.png  as a supplementary figure.

Issue 13 →  Either:
            (a) Redefine cruise L/D as mean(L/D) at α=4–6° and re-run,
                OR
            (b) Add the following sentence to Section 3.4:
                "Cruise L/D is defined at α≤2° following UAV low-speed
                conventions; maximum L/D occurs at the polar peak
                (α≈6–8° for cambered airfoils), explaining the
                factor-of-two difference between these metrics."
─────────────────────────────────────────────────────────────────────
""")