"""fig2.py - Fig 2a-f from Processed/fig2_stats. python fig2.py"""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm

SRC = Path(r"C:\Haadia AI Project\Processed\fig2_stats")
OUT = Path(r"C:\Haadia AI Project\Graphs"); OUT.mkdir(exist_ok=True)
TEAL, ORANGE, RED = "#5BC0BE", "#F28E2B", "#E8453C"

Z = [np.load(f) for f in sorted(SRC.glob("vin*.npz"))]
g = lambda k: np.array([float(z[k]) for z in Z])
mil, dur, inten = g("mileage"), g("duration"), g("intensity")
spd = g("avg_speed_all") if "avg_speed_all" in Z[0].files else g("avg_speed")
s0 = np.concatenate([z["soc_start"] for z in Z]); s1 = np.concatenate([z["soc_end"] for z in Z])
if "run_gain" in Z[0].files:
    rg = np.concatenate([z["run_gain"] for z in Z]); rs = np.concatenate([z["run_s"] for z in Z])
    keep = (rg >= 5) & (rs >= 300)          # [ASSUMPTION] real charge: >=5 SOC points gained, >=5 min
    print(f"charges {len(s0):,} -> {keep.sum():,} | start>=95% before {np.mean(s0>=95):.1%}, after {np.mean(s0[keep]>=95):.1%}")
    s0, s1 = s0[keep], s1[keep]
else:
    print("WARNING: run_gain missing, re-run fleet_stats.py with the new fields")
print(f"vehicles {len(Z)} | charges {len(s0):,}")
for n, a in [("mileage", mil), ("duration", dur), ("intensity", inten), ("speed", spd)]:
    a = a[~np.isnan(a)]; print(f"{n:9s} n={len(a)} min {a.min():.1f} med {np.median(a):.1f} max {a.max():.1f}")

fig, ax = plt.subplots(2, 4, figsize=(16, 7))
def hist(a, x, data, xl, bins):
    d = data[~np.isnan(data)]
    a.hist(d, bins=bins, color=TEAL, edgecolor="k", linewidth=.8)
    a.set_xlabel(xl); a.set_ylabel("Count")
for a, d, xl, b in zip(ax[0], (mil, dur, inten, spd),
        ("Mileage (km)", "Duration (days)", "Use intensity (km/day)", "Average speed (km/h)"), (25, 25, 25, 25)):
    hist(a, None, d, xl, b)
for a, t in zip(ax[0], "abcd"): a.set_title(f"({t})", loc="left")

# e: start SOC + fitted curve and dashed line at the mean [ASSUMPTION: Gaussian fit, mean marker]
a = ax[1, 0]; c, e, _ = a.hist(s0, bins=50, range=(0, 100), color=TEAL, edgecolor="k", linewidth=.5)
mu, sd = s0.mean(), s0.std(); x = np.linspace(0, 100, 300)
a.plot(x, norm.pdf(x, mu, sd) * len(s0) * (e[1] - e[0]), color=RED, lw=2)
a.axvline(mu, color=ORANGE, ls="--", lw=3); a.set_xlabel("Charge start SOC (%)"); a.set_ylabel("Count")
a.set_title("(e)", loc="left")
a = ax[1, 1]; a.hist(s1, bins=20, range=(0, 100), color=TEAL, edgecolor="k", linewidth=.5)
a.set_xlabel("Charge end SOC (%)"); a.set_ylabel("Count"); a.set_title("(f)", loc="left")
for a in ax[1, 2:]: a.axis("off")          # g-j added later
for a in ax.flat: a.ticklabel_format(axis="y", style="sci", scilimits=(0, 4)) if a.axison else None
plt.tight_layout(); plt.savefig(OUT / "fig2a_f.png", dpi=200); print("saved", OUT / "fig2a_f.png")