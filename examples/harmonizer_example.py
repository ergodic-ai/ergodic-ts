"""Harmonizer — example script.

Demonstrates two reconciliation scenarios:
  1. Hierarchical (additive) — National = East + West
  2. Price + elasticity — ASP × Qty = Revenue with demand elasticity

Run::

    uv run python examples/harmonizer_example.py
"""

from __future__ import annotations

import numpy as np

from ergodicts import (
    AdditiveConstraint,
    DependencyGraph,
    ElasticityConstraint,
    ForecastBelief,
    Harmonizer,
    ModelKey,
    PriceConstraint,
)

# ============================================================
# Example 1:  Hierarchical harmonization (analytical)
# ============================================================
print("=" * 60)
print("Example 1: Hierarchical Harmonization")
print("=" * 60)

# --- hierarchy ---
national = ModelKey(("REGION",), ("NATIONAL",))
east = ModelKey(("REGION",), ("EAST",))
west = ModelKey(("REGION",), ("WEST",))

hierarchy = DependencyGraph()
hierarchy.add(national, east)
hierarchy.add(national, west)

# --- beliefs (intentionally inconsistent: 200 ≠ 120 + 90) ---
beliefs_hier = {
    national: ForecastBelief(
        distribution="normal",
        params={
            "loc": np.array([200.0, 210.0, 220.0]),
            "scale": np.array([10.0, 10.0, 10.0]),
        },
    ),
    east: ForecastBelief(
        distribution="normal",
        params={
            "loc": np.array([120.0, 125.0, 130.0]),
            "scale": np.array([5.0, 5.0, 5.0]),
        },
    ),
    west: ForecastBelief(
        distribution="normal",
        params={
            "loc": np.array([90.0, 95.0, 100.0]),
            "scale": np.array([8.0, 8.0, 8.0]),
        },
    ),
}

harmonizer = Harmonizer(hierarchy=hierarchy)
result = harmonizer.harmonize(
    beliefs_hier,
    method="analytical",
    num_samples=1000,
    rng_seed=42,
)

means = result.mean()
print("\nReconciled means:")
for key in [national, east, west]:
    print(f"  {key.label:20s}  {np.array2string(means[key], precision=1)}")

gap = means[national] - (means[east] + means[west])
print(f"\n  National - (East + West) = {np.array2string(gap, precision=4)}")
print(f"  Max |gap| = {np.abs(gap).max():.6f}  (should be ~0)")

print("\nSummary DataFrame:")
print(result.summary().to_string(index=False))

# ============================================================
# Example 2:  Price + elasticity harmonization (MCMC)
# ============================================================
print("\n" + "=" * 60)
print("Example 2: Price + Elasticity Harmonization")
print("=" * 60)

asp = ModelKey(("METRIC",), ("ASP",))
qty = ModelKey(("METRIC",), ("QTY",))
rev = ModelKey(("METRIC",), ("REVENUE",))

T = 6
beliefs_price = {
    asp: ForecastBelief(
        distribution="normal",
        params={
            "loc": np.array([50.0, 51.0, 52.0, 53.0, 54.0, 55.0]),
            "scale": np.array([2.0] * T),
        },
    ),
    qty: ForecastBelief(
        distribution="normal",
        params={
            "loc": np.array([1000.0, 990.0, 980.0, 970.0, 960.0, 950.0]),
            "scale": np.array([50.0] * T),
        },
    ),
    # Revenue forecast: intentionally inconsistent with ASP × Qty.
    # ASP × Qty at t=0 is ~50k, but our revenue model says ~52k.
    # The harmonizer will pull these toward agreement.
    rev: ForecastBelief(
        distribution="normal",
        params={
            "loc": np.array([52000.0, 52000.0, 52000.0, 52000.0, 52000.0, 52000.0]),
            "scale": np.array([3000.0] * T),
        },
    ),
}

print("\nBefore harmonization:")
print(f"  ASP × Qty = {beliefs_price[asp].params['loc'] * beliefs_price[qty].params['loc']}")
print(f"  Revenue   = {beliefs_price[rev].params['loc']}")
print(f"  Gap       = {beliefs_price[asp].params['loc'] * beliefs_price[qty].params['loc'] - beliefs_price[rev].params['loc']}")

harmonizer = Harmonizer(
    constraints=[
        PriceConstraint(asp_key=asp, qty_key=qty, dollar_key=rev),
        ElasticityConstraint(asp_key=asp, qty_key=qty, elasticity_prior=(-1.0, 0.5)),
    ],
)

result = harmonizer.harmonize(
    beliefs_price,
    method="mcmc",
    num_samples=500,
    lambda_price=10.0,
    lambda_elasticity=0.5,
    mcmc_kwargs={"num_warmup": 300, "num_chains": 1},
    rng_seed=42,
)

means = result.mean()
print("\nAfter harmonization:")
print(f"  ASP       = {np.array2string(means[asp], precision=1)}")
print(f"  Qty       = {np.array2string(means[qty], precision=1)}")
print(f"  Revenue   = {np.array2string(means[rev], precision=1)}")
print(f"  ASP × Qty = {np.array2string(means[asp] * means[qty], precision=1)}")
print(f"  Gap       = {np.array2string(means[asp] * means[qty] - means[rev], precision=1)}")

print("\nConstraint violations:")
for name, v in result.constraint_violations.items():
    print(f"  {name}: mean |residual| = {v.mean():.2f}")

# --- optional: plot if matplotlib is available ---
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    labels = {asp: "ASP ($)", qty: "Quantity", rev: "Revenue ($)"}

    for ax, key in zip(axes, [asp, qty, rev]):
        original = beliefs_price[key].params["loc"]
        reconciled = means[key]
        samples = result.samples[key]
        x = np.arange(T)
        q10 = np.quantile(samples, 0.1, axis=0)
        q90 = np.quantile(samples, 0.9, axis=0)
        ax.fill_between(x, q10, q90, alpha=0.2, color="C1", label="80% CI")
        ax.plot(x, original, "k--", label="Original belief")
        ax.plot(x, reconciled, "C1-", linewidth=2, label="Reconciled")
        ax.set_title(labels[key])
        ax.set_xlabel("Month")
        ax.legend(fontsize=8)

    plt.suptitle("Price + Elasticity Harmonization")
    plt.tight_layout()
    plt.savefig("examples/harmonizer_example_results.png", dpi=150)
    print("\nPlot saved to examples/harmonizer_example_results.png")
except ImportError:
    print("\nInstall matplotlib to generate plots.")
