"""fig3.py - Fig 3a-f. python fig3.py [--vin vin1]"""
import argparse, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde
from extract import read_vehicle, find_segments, extract_hi, RAW_DIR, VMAX_, VMIN_, CUR_, V_HI, log

FLEET = Path(r"C:\Haadia AI Project\Processed\fleet")
OUT = Path(r"C:\Haadia AI Project\Graphs"); OUT.mkdir(exist_ok=True)
PUR, PINK, GRN, RED = "#7E57C2", "#F06292", "#2E8B2E", "#E8453C"
ap = argparse.ArgumentParser(); ap.add_argument("--vin", default="vin1"); a = ap.parse_args()

# ---- pick one charge with valid HIs and a current switch inside the window [ASSUMPTION: >=30% current change]
A, cand, _ = read_vehicle(RAW_DIR / f"{a.vin}.rar")
best = None
for sg in find_segments(A, {}):
    res, why = extract_hi(sg, A, cand)
    if res is None:
        continue
    s, e = sg["s"], sg["e"]
    vm, cur = A[s:e + 1, VMAX_], -A[s:e + 1, CUR_]
    i1 = np.flatnonzero(vm >= V_HI[0])[0]; i2 = i1 + np.flatnonzero(vm[i1:] >= V_HI[-1])[0]
    sw = abs(cur[i2] - cur[i1]) / max(cur[i1], 1e-6)
    if best is None or sw > best[0]:
        best = (sw, sg, res, i1, i2)
    if sw >= 0.3:
        break
sw, sg, (vmap, seq, pts), i1, i2 = best
s, e = sg["s"], sg["e"]
log(f"sample: rows {s}-{e}, SOH {sg['soh']:.3f}, start SOC {sg['soc_start']:.0f}%, current change in window {sw:.0%}")
vm, vn, cur = A[s:e + 1, VMAX_], A[s:e + 1, VMIN_], -A[s:e + 1, CUR_]
x = np.arange(len(vm))

fig, ax = plt.subplots(2, 3, figsize=(16, 8))
for a_, l in zip(ax.flat, "abcdef"):
    a_.set_title(f"({l})", loc="left")

# a
A_ = ax[0, 0]; A_.plot(x, vm, color=PUR); A_.set_xlabel("Sample"); A_.set_ylabel("Voltage (V)", color=PUR)
B_ = A_.twinx(); B_.plot(x, cur, color=PINK, ls="--"); B_.set_ylabel("Current (A)", color=PINK)
for i in (i1, i2): A_.axvline(i, color="g", ls="--", lw=1)
A_.annotate("Start\ncharging", (0, vm[0]), (len(x) * .05, vm[0] + .1), arrowprops=dict(arrowstyle="->"))
k = np.argmax(np.abs(np.diff(cur))); A_.annotate("Current\nswitch", (k, vm[k]), (k, vm[k] - .15), arrowprops=dict(arrowstyle="->"))

# b: Q_HI curves of one vehicle, coloured by SOH
z = np.load(FLEET / f"{a.vin}.npz"); ok = z["outlier"] == 0
Q, soh = z["seqs"][ok, 0], z["soh"][ok]
sel = np.random.RandomState(0).permutation(len(Q))[:80]; sel = sel[np.argsort(soh[sel])]
nm = Normalize(soh[sel].min(), soh[sel].max()); cm = plt.get_cmap("Blues")
for i in sel: ax[0, 1].plot(V_HI * 1000, Q[i], color=cm(.3 + .7 * nm(soh[i])), lw=1)
ax[0, 1].set_xlabel("Voltage (mV)"); ax[0, 1].set_ylabel("Increment capacity (Ah)")
fig.colorbar(ScalarMappable(nm, cm), ax=ax[0, 1], label="SOH")

# c
ax[0, 2].plot(V_HI * 1000, seq[0], color=GRN); ax[0, 2].set_xlabel("Voltage (mV)")
ax[0, 2].set_ylabel("Increment capacity (Ah)", color=GRN)
c2 = ax[0, 2].twinx(); c2.plot(V_HI * 1000, seq[1], color=RED); c2.set_ylabel("Temperature (°C)", color=RED)

# d
ax[1, 0].plot(x, vm, color=PUR, label="Max cell voltage"); ax[1, 0].plot(x, vn, color=PUR, ls="--", label="Min cell voltage")
ax[1, 0].set_xlabel("Sample"); ax[1, 0].set_ylabel("Cell voltage (V)"); ax[1, 0].legend(fontsize=8)
d2 = ax[1, 0].twinx(); d2.plot(x, vm - vn, color=PINK); d2.set_ylabel("Cell voltage range (V)", color=PINK)

# e: rows = 96 uniformly spaced time steps in the window; y-axis labelled with the nominal 3.90-4.05 V [ASSUMPTION]
im = ax[1, 1].imshow(vmap, origin="lower", aspect="auto", cmap="Greens", extent=[1, 96, 3.90, 4.05])
ax[1, 1].set_xlabel("Cell number"); ax[1, 1].set_ylabel("Voltage (V)"); fig.colorbar(im, ax=ax[1, 1], label="Cell voltage (V)")

# f: min-max normalised points, pooled over the fleet [ASSUMPTION: normalisation over all samples]
P = np.load(FLEET / "train_cache.npz")["p"].astype(np.float64)
P = (P - P.min(0)) / (np.ptp(P, 0) + 1e-9)
sub = np.random.RandomState(0).permutation(len(P))[:5000]; xs = np.linspace(-.2, 1.2, 400)
for j in range(P.shape[1]):
    try:
        y = gaussian_kde(P[sub, j])(xs); ax[1, 2].plot(xs, y, lw=1); ax[1, 2].fill_between(xs, y, alpha=.3)
    except Exception:
        pass                                   # constant feature -> singular KDE
ax[1, 2].set_xlabel("Point features (normalized)"); ax[1, 2].set_ylabel("Density")

plt.tight_layout(); plt.savefig(OUT / "fig3.png", dpi=200); log(f"saved {OUT / 'fig3.png'}")