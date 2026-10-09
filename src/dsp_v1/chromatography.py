"""1D transport-dispersive, Langmuir + linear-driving-force breakthrough model.

Single binding solute; the model represents loading, not elution, impurity
clearance or Protein A chemistry. Units: cm, mL, min, mg.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ChromatographyParameters:
    column_length_cm: float = 10.0
    column_diameter_cm: float = 1.0
    bed_porosity: float = 0.40
    flow_ml_min: float = 1.0
    feed_mg_ml: float = 1.0
    capacity_mg_ml_solid: float = 50.0
    langmuir_affinity_ml_mg: float = 2.0
    ldf_rate_per_min: float = 0.035
    dispersion_cm2_min: float = 0.03
    cells: int = 60

    def __post_init__(self) -> None:
        for name in (
            "column_length_cm", "column_diameter_cm", "flow_ml_min", "feed_mg_ml",
            "capacity_mg_ml_solid", "langmuir_affinity_ml_mg", "ldf_rate_per_min",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if not 0 < self.bed_porosity < 1:
            raise ValueError("bed_porosity must be strictly between 0 and 1")
        if not np.isfinite(self.dispersion_cm2_min) or self.dispersion_cm2_min < 0:
            raise ValueError("dispersion_cm2_min must be finite and >= 0")
        if not isinstance(self.cells, int) or self.cells < 5:
            raise ValueError("cells must be an integer >= 5")

    @property
    def cross_section_cm2(self) -> float:
        return np.pi * (self.column_diameter_cm / 2) ** 2

    @property
    def bed_volume_ml(self) -> float:
        return self.cross_section_cm2 * self.column_length_cm

    @property
    def superficial_velocity_cm_min(self) -> float:
        return self.flow_ml_min / self.cross_section_cm2

    @property
    def void_residence_time_min(self) -> float:
        """Ideal hydraulic void volume divided by volumetric flow; no tubing hold-up."""
        return self.bed_porosity * self.bed_volume_ml / self.flow_ml_min

    @property
    def residence_time_min(self) -> float:
        """Empty-bed residence time = bed volume / flow."""
        return self.bed_volume_ml / self.flow_ml_min

    @property
    def equilibrium_capacity_mg_ml_solid(self) -> float:
        c = self.feed_mg_ml
        return (self.capacity_mg_ml_solid * self.langmuir_affinity_ml_mg * c
                / (1 + self.langmuir_affinity_ml_mg * c))

    def suggested_end_time_min(self) -> float:
        estimated_mass_mg = self.bed_volume_ml * (
            self.bed_porosity * self.feed_mg_ml
            + (1 - self.bed_porosity) * self.equilibrium_capacity_mg_ml_solid
        )
        return max(45.0, 1.8 * estimated_mass_mg / (self.flow_ml_min * self.feed_mg_ml))


@dataclass
class ChromatographyResult:
    params: ChromatographyParameters
    time_min: np.ndarray
    outlet_mg_ml: np.ndarray
    dissolved_mg: np.ndarray
    adsorbed_mg: np.ndarray
    outlet_cumulative_mg: np.ndarray

    @property
    def outlet_fraction(self) -> np.ndarray:
        return self.outlet_mg_ml / self.params.feed_mg_ml

    @property
    def loaded_cv(self) -> np.ndarray:
        return self.time_min * self.params.flow_ml_min / self.params.bed_volume_ml

    @property
    def loaded_mg(self) -> np.ndarray:
        return self.time_min * self.params.flow_ml_min * self.params.feed_mg_ml

    @property
    def mass_balance_relative_error(self) -> float:
        if self.loaded_mg[-1] == 0:
            return 0.0
        balance = self.loaded_mg - (self.outlet_cumulative_mg + self.dissolved_mg + self.adsorbed_mg)
        return float(np.max(np.abs(balance)) / self.loaded_mg[-1])

    def threshold_time_min(self, fraction: float = 0.10) -> float:
        if not 0 < fraction < 1:
            raise ValueError("fraction must be in (0, 1)")
        crossed = np.flatnonzero(self.outlet_fraction >= fraction)
        if not len(crossed):
            raise ValueError("Simulation ended before requested breakthrough threshold")
        idx = int(crossed[0])
        if idx == 0:
            return float(self.time_min[0])
        return float(np.interp(
            fraction,
            self.outlet_fraction[idx - 1:idx + 1],
            self.time_min[idx - 1:idx + 1],
        ))

    def metrics(self) -> dict[str, float]:
        """At 10% breakthrough report 3 separate, explicitly defined measures.

        - loaded_capacity: gross protein loaded through column, per mL packed bed;
        - void_corrected_dbc10: conventional simple void correction using
          eps*bed_volume (NOT system dead volume or measured tracer delay);
        - uptake_at_10: simulated mass adsorbed in solid phase per mL packed bed.

        These measures are NOT interchangeable and should not be confused with
        actual experimentally determined DBC10.
        """
        t10 = self.threshold_time_min(0.1)
        t50 = self.threshold_time_min(0.5)
        p = self.params
        cv10 = t10 / p.residence_time_min
        return {
            "flow_ml_min": p.flow_ml_min,
            "empty_bed_residence_min": p.residence_time_min,
            "breakthrough_10_min": t10,
            "breakthrough_10_cv": cv10,
            "breakthrough_50_min": t50,
            "gross_loaded_at_10_mg_ml_bed": cv10 * p.feed_mg_ml,
            "dbc10_void_corrected_mg_ml_bed": max(0.0, (cv10 - p.bed_porosity) * p.feed_mg_ml),
            "simulated_adsorbed_at_10_mg_ml_bed": float(np.interp(t10, self.time_min, self.adsorbed_mg)) / p.bed_volume_ml,
            "mass_balance_max_error_pct": 100 * self.mass_balance_relative_error,
        }


def simulate_chromatography(
    params: ChromatographyParameters | None = None,
    *, end_time_min: float | None = None,
    samples: int = 751,
) -> ChromatographyResult:
    """Finite-volume 1D adsorption breakthrough calculation.

    Conservative bed-volume balance:
      d/dt[eps*c + (1-eps)*q] = -d/dz[u*c - eps*Dax*dc/dz]
      dq/dt = k_LDF * (qeq(c) - q)
      qeq = qmax*K*c / (1 + K*c)

    Upwind convective face flux, central diffusive face flux, specified
    inlet total flux u*Cfeed (Danckwerts type) and zero-gradient outlet.
    """
    p = params or ChromatographyParameters()
    if end_time_min is None:
        end_time_min = p.suggested_end_time_min()
    if not np.isfinite(end_time_min) or end_time_min <= 0:
        raise ValueError("end_time_min must be > 0")
    if samples < 5:
        raise ValueError("samples must be >= 5")
    n = p.cells
    dz = p.column_length_cm / n
    u = p.superficial_velocity_cm_min
    eps = p.bed_porosity
    effective_dispersion = eps * p.dispersion_cm2_min / dz

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        c = y[:n]
        q = y[n:2*n]
        eq = p.capacity_mg_ml_solid * (p.langmuir_affinity_ml_mg * c) / (1 + p.langmuir_affinity_ml_mg * c)
        qdot = p.ldf_rate_per_min * (eq - q)
        # Face fluxes, in mg / (cm^2 min); gradient at last face is zero.
        faces = np.empty(n + 1)
        faces[0] = u * p.feed_mg_ml
        faces[1:-1] = u * c[:-1] - effective_dispersion * (c[1:] - c[:-1])
        faces[-1] = u * c[-1]
        cdot = (-(faces[1:] - faces[:-1]) / dz - (1 - eps) * qdot) / eps
        moutdot = p.flow_ml_min * c[-1]
        return np.concatenate((cdot, qdot, np.array([moutdot])))

    y0 = np.zeros(2*n + 1)
    times = np.linspace(0, end_time_min, samples)
    solution = solve_ivp(
        rhs, (0, end_time_min), y0, method="BDF", t_eval=times,
        rtol=1e-7, atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"Chromatography solver failed: {solution.message}")
    c = solution.y[:n]
    q = solution.y[n:2*n]
    cell_volume = p.bed_volume_ml / n
    dissolved = np.sum(c, axis=0) * eps * cell_volume
    adsorbed = np.sum(q, axis=0) * (1-eps) * cell_volume
    result = ChromatographyResult(
        p, times, c[-1], dissolved, adsorbed, solution.y[-1]
    )
    # Mass balance evaluated numerically with a state-integrated outlet flux.
    if result.mass_balance_relative_error > 1e-4:
        raise RuntimeError(f"Conservation check failed: {result.mass_balance_relative_error:.4g}")
    return result
