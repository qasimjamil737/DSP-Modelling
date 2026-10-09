"""Generate synthetic chromatography and UF/DF examples (CSV, figures, metrics)."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biopharma_dsp import (
    ChromatographyParameters, UFDFParameters,
    simulate_chromatography, simulate_ufdf,
)


def make_figure(ax):
    ax.grid(True, alpha=0.18)
    for s in ("right", "top"):
        ax.spines[s].set_visible(False)


def run(out: Path, config: dict) -> None:
    out.mkdir(exist_ok=True, parents=True)
    chrom = ChromatographyParameters(**config.get("chromatography", {}))
    ufdf = UFDFParameters(**config.get("ufdf", {}))
    baseline = simulate_chromatography(chrom)

    metrics: dict = {"chromatography": baseline.metrics()}
    print("Chromatography baseline:", json.dumps(metrics["chromatography"], indent=2))

    pd.DataFrame({
        "time_min": baseline.time_min,
        "loaded_cv": baseline.loaded_cv,
        "outlet_fraction_c_over_c0": baseline.outlet_fraction,
        "cumulative_loaded_mg": baseline.loaded_mg,
        "cumulative_eluted_mg": baseline.outlet_cumulative_mg,
        "liquid_phase_mg": baseline.dissolved_mg,
        "stationary_phase_mg": baseline.adsorbed_mg,
    }).to_csv(out / "chromatography_baseline.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(baseline.loaded_cv, baseline.outlet_fraction, label="Mechanistic model", linewidth=2)
    cv10 = baseline.metrics()["breakthrough_10_cv"]
    ax.axhline(0.1, color="#546574", linestyle="--", linewidth=1, label="10% breakthrough")
    ax.axvline(cv10, color="#546574", linestyle=":", linewidth=1, label=f"CV10={cv10:.1f}")
    ax.set(xlabel="Loaded packed-bed column volumes [CV]", ylabel="Outlet / feed concentration [–]",
           title="Protein chromatography: simulated breakthrough", ylim=(0, 1.02))
    ax.legend()
    make_figure(ax)
    fig.tight_layout()
    fig.savefig(out / "chromatography_breakthrough.png", dpi=180)
    plt.close(fig)

    # Numerical-resolution check: conservative discretization is not grid-independent.
    grid_sizes = config.get("grid_cells", [30, 60, 120])
    if not isinstance(grid_sizes, list) or any(not isinstance(n, int) or n < 5 for n in grid_sizes):
        raise ValueError("grid_cells must be a list of integers >= 5")
    grid_table = []
    for n in grid_sizes:
        numerical = simulate_chromatography(replace(chrom, cells=n), samples=751)
        mm = numerical.metrics()
        grid_table.append({"cells": n,
                           "cv10": mm["breakthrough_10_cv"],
                           "dbc10_void_corrected_mg_ml_bed": mm["dbc10_void_corrected_mg_ml_bed"],
                           "mass_balance_max_error_pct": mm["mass_balance_max_error_pct"]})
    pd.DataFrame(grid_table).to_csv(out / "chromatography_grid_refinement.csv", index=False)
    metrics["numerical_grid_check"] = grid_table

    # One-factor-at-a-time residence-time analysis; all other parameters fixed.
    multipliers = config.get("flow_multipliers", [0.5, 0.75, 1.0, 1.5, 2.0])
    if not isinstance(multipliers, list) or not multipliers or any(not isinstance(a, (int, float)) or a <= 0 for a in multipliers):
        raise ValueError("flow_multipliers must be a list of positive numbers")
    sensitivity = []
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for factor in multipliers:
        pp = replace(chrom, flow_ml_min=chrom.flow_ml_min*factor)
        trial = simulate_chromatography(pp)
        row = trial.metrics()
        row["flow_multiplier"] = factor
        sensitivity.append(row)
        ax.plot(trial.loaded_cv, trial.outlet_fraction, label=f"{pp.flow_ml_min:g} mL/min")
    ax.axhline(0.1, color="#546574", linestyle="--", linewidth=1)
    ax.set(xlabel="Loaded column volumes [CV]", ylabel="Outlet / feed concentration [–]",
           title="Influence of flow rate on breakthrough", ylim=(0, 1.02))
    ax.legend(title="Flow")
    make_figure(ax)
    fig.tight_layout()
    fig.savefig(out / "chromatography_flow_sensitivity.png", dpi=180)
    plt.close(fig)
    sensitivity_df = pd.DataFrame(sensitivity)
    sensitivity_df.to_csv(out / "chromatography_flow_sensitivity.csv", index=False)
    metrics["flow_sensitivity"] = sensitivity

    # Compact two-factor model-screening grid (illustrative, not an experimental DoE).
    ldf_multipliers = config.get("ldf_multipliers", [0.5, 1.0, 2.0])
    if not isinstance(ldf_multipliers, list) or not ldf_multipliers or any(not isinstance(a, (int, float)) or a <= 0 for a in ldf_multipliers):
        raise ValueError("ldf_multipliers must be a list of positive numbers")
    screening = []
    for ldf_factor in ldf_multipliers:
        for flow_factor in multipliers:
            pp = replace(chrom,
                         flow_ml_min=chrom.flow_ml_min*flow_factor,
                         ldf_rate_per_min=chrom.ldf_rate_per_min*ldf_factor)
            trial = simulate_chromatography(pp, samples=501)
            m = trial.metrics()
            screening.append({
                "flow_ml_min": pp.flow_ml_min,
                "ldf_rate_per_min": pp.ldf_rate_per_min,
                "flow_multiplier": flow_factor,
                "ldf_multiplier": ldf_factor,
                "dbc10_void_corrected_mg_ml_bed": m["dbc10_void_corrected_mg_ml_bed"],
                "cv10": m["breakthrough_10_cv"],
            })
    screen_df = pd.DataFrame(screening)
    screen_df.to_csv(out / "chromatography_screening_grid.csv", index=False)
    heat = screen_df.pivot(index="ldf_multiplier", columns="flow_multiplier", values="dbc10_void_corrected_mg_ml_bed")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    img = ax.imshow(heat.to_numpy(), aspect="auto", cmap="viridis")
    ax.set(xticks=np.arange(len(heat.columns)), yticks=np.arange(len(heat.index)),
           xticklabels=[f"{v:g}" for v in heat.columns],
           yticklabels=[f"{v:g}" for v in heat.index],
           xlabel="Flow-rate multiplier [–]", ylabel="LDF-rate multiplier [–]",
           title="Model-screening grid: void-corrected DBC10")
    for iy in range(len(heat.index)):
        for ix in range(len(heat.columns)):
            ax.text(ix, iy, f"{heat.iloc[iy, ix]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(img, ax=ax, label="DBC10 [mg/mL bed]")
    fig.tight_layout()
    fig.savefig(out / "chromatography_screening_heatmap.png", dpi=180)
    plt.close(fig)
    metrics["screening_grid"] = screening

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    df = sensitivity_df.sort_values("empty_bed_residence_min")
    ax.plot(df["empty_bed_residence_min"], df["dbc10_void_corrected_mg_ml_bed"], "o-", linewidth=2)
    ax.set(xlabel="Empty-bed residence time [min]", ylabel="Void-corrected DBC10 [mg/mL bed]",
           title="Residence-time sensitivity: synthetic parameters")
    make_figure(ax)
    fig.tight_layout()
    fig.savefig(out / "chromatography_dbc_vs_residence.png", dpi=180)
    plt.close(fig)

    tff = simulate_ufdf(ufdf)
    metrics["ufdf"] = tff.metrics()
    print("UF/DF baseline:", json.dumps(metrics["ufdf"], indent=2))
    pd.DataFrame({
        "time_h": tff.time_h,
        "stage": tff.stages,
        "volume_l": tff.volume_l,
        "protein_mass_g": tff.protein_mass_g,
        "impurity_mass_g": tff.impurity_mass_g,
        "protein_g_l": tff.protein_mass_g/tff.volume_l,
        "impurity_g_l": tff.impurity_mass_g/tff.volume_l,
    }).to_csv(out / "ufdf_timeseries.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.7))
    ax.plot(tff.time_h, 100*tff.protein_mass_g/tff.protein_mass_g[0], label="Protein retained [%]", linewidth=2)
    ax.plot(tff.time_h, 100*tff.impurity_mass_g/tff.impurity_mass_g[0], label="Impurity remaining [%]", linewidth=2)
    uf_time = (ufdf.starting_volume_l-ufdf.target_volume_l)/(ufdf.flux_l_m2_h*ufdf.membrane_area_m2)
    ax.axvline(uf_time, linestyle="--", color="#546574", label="UF → DF transition")
    ax.set(xlabel="Process time [h]", ylabel="Starting mass remaining [%]",
           title="Idealized UF concentration + constant-volume diafiltration", ylim=(0, 103))
    ax.legend()
    make_figure(ax)
    fig.tight_layout()
    fig.savefig(out / "ufdf_mass_retention.png", dpi=180)
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Outputs:", out.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "examples" / "config.json")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.out, raw)
