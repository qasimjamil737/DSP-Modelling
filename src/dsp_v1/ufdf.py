"""Idealized well-mixed ultrafiltration and constant-volume diafiltration balances.

This is an intentionally simplified stage calculator (not a membrane transport,
fouling, or product quality predictor). Flux is constant; sieving is constant.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class UFDFParameters:
    starting_volume_l: float = 5.0
    target_volume_l: float = 1.0
    protein_g_l: float = 5.0
    impurity_g_l: float = 1.0
    membrane_area_m2: float = 0.10
    flux_l_m2_h: float = 30.0
    protein_sieving: float = 0.001
    impurity_sieving: float = 0.90
    diafiltration_volumes: float = 5.0

    def __post_init__(self) -> None:
        for name in (
            "starting_volume_l", "target_volume_l", "protein_g_l", "impurity_g_l",
            "membrane_area_m2", "flux_l_m2_h",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.target_volume_l > self.starting_volume_l:
            raise ValueError("target_volume_l must not exceed starting_volume_l")
        if not 0 <= self.protein_sieving <= 1 or not 0 <= self.impurity_sieving <= 1:
            raise ValueError("sieving coefficients must be between 0 and 1")
        if not np.isfinite(self.diafiltration_volumes) or self.diafiltration_volumes < 0:
            raise ValueError("diafiltration_volumes must be finite and >= 0")


@dataclass
class UFDFResult:
    params: UFDFParameters
    time_h: np.ndarray
    volume_l: np.ndarray
    protein_mass_g: np.ndarray
    impurity_mass_g: np.ndarray
    stages: np.ndarray
    process_time_h: float

    def metrics(self) -> dict[str, float]:
        p = self.params
        initial_protein_g = p.starting_volume_l * p.protein_g_l
        initial_impurity_g = p.starting_volume_l * p.impurity_g_l
        return {
            "processing_time_h": self.process_time_h,
            "final_volume_l": float(self.volume_l[-1]),
            "protein_recovery_pct": 100 * float(self.protein_mass_g[-1]) / initial_protein_g,
            "impurity_removal_pct": 100 * (1 - float(self.impurity_mass_g[-1]) / initial_impurity_g),
            "final_protein_g_l": float(self.protein_mass_g[-1]) / float(self.volume_l[-1]),
            "final_impurity_g_l": float(self.impurity_mass_g[-1]) / float(self.volume_l[-1]),
        }


def simulate_ufdf(p: UFDFParameters | None = None, *, points_per_stage: int = 151) -> UFDFResult:
    """Analytic mass-balance solution for UF concentration followed by DF.

    UF: M(V)/M0 = (V/V0)^S for each solute, with dV/dt=-A*J.
    DF: M(N)/M(UF_end) = exp(-S*N), with N=permeate volume/Vtarget,
        maintained at constant retentate volume by matching fresh buffer inflow.
    """
    p = p or UFDFParameters()
    if points_per_stage < 2:
        raise ValueError("points_per_stage must be >= 2")
    permeate_l_h = p.membrane_area_m2 * p.flux_l_m2_h
    uf_time = (p.starting_volume_l - p.target_volume_l) / permeate_l_h
    t_uf = np.linspace(0, uf_time, points_per_stage)
    vol_uf = p.starting_volume_l - permeate_l_h * t_uf
    mp_uf = p.starting_volume_l * p.protein_g_l * (vol_uf / p.starting_volume_l) ** p.protein_sieving
    mi_uf = p.starting_volume_l * p.impurity_g_l * (vol_uf / p.starting_volume_l) ** p.impurity_sieving

    n_dv = np.linspace(0, p.diafiltration_volumes, points_per_stage)
    t_df = uf_time + n_dv * p.target_volume_l / permeate_l_h
    vol_df = np.full_like(t_df, p.target_volume_l)
    mp_df = mp_uf[-1] * np.exp(-p.protein_sieving * n_dv)
    mi_df = mi_uf[-1] * np.exp(-p.impurity_sieving * n_dv)

    return UFDFResult(
        p,
        np.r_[t_uf, t_df[1:]],
        np.r_[vol_uf, vol_df[1:]],
        np.r_[mp_uf, mp_df[1:]],
        np.r_[mi_uf, mi_df[1:]],
        np.array(["Concentration"]*len(t_uf)+["Diafiltration"]*(len(t_df)-1)),
        float(t_df[-1]),
    )
