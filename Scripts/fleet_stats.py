"""fleet_stats.py - per-vehicle stats for Fig 2a-f. Resumable. python fleet_stats.py [--n 5]"""
import argparse, time
import numpy as np, pandas as pd
from pathlib import Path
from extract import open_csv, RAW_DIR, log

OUT = Path(r"C:\Haadia AI Project\Processed\fig2_stats")
COLS = ["terminaltime", "soc", "speed", "totalodometer", "chargestatus"]


def one(p):
    parts = []
    with open_csv(p) as (f, _):
        for ch in pd.read_csv(f, encoding="utf-8-sig", chunksize=500_000, usecols=COLS):
            parts.append(ch.apply(pd.to_numeric, errors="coerce").to_numpy(np.float64))
    X = np.vstack(parts); X = X[~np.isnan(X[:, 0])]
    X = X[np.argsort(X[:, 0], kind="stable")]
    X = X[np.r_[True, np.diff(X[:, 0]) > 0]]
    t, soc, spd, odo, cs = X.T
    okodo = (odo >= 0) & (odo < 2e6)                         # drop sentinel odometer values
    mileage = odo[okodo].max() if okodo.any() else np.nan
    dur = (t[-1] - t[0]) / 86400
    mv = (spd > 0) & (spd < 200)
    okspd = (spd >= 0) & (spd < 200)
    chg = cs == 1
    brk = np.r_[True, chg[1:] != chg[:-1]] | np.r_[True, np.diff(t) > 600]   # new charge run
    st = np.flatnonzero(chg & brk)
    en = np.flatnonzero(chg & np.r_[brk[1:], True])
    s0, s1 = soc[st], soc[en]
    rg = s1 - s0
    rs = t[en] - t[st]
    ok = (s0 >= 0) & (s0 <= 100) & (s1 >= 0) & (s1 <= 100)
    return dict(mileage=mileage, duration=dur, intensity=mileage / dur if dur > 0 else np.nan,
                avg_speed=spd[mv].mean() if mv.any() else np.nan,
                avg_speed_all=spd[okspd].mean() if okspd.any() else np.nan,
                soc_start=s0[ok], soc_end=s1[ok],
                run_gain=rg[ok], run_s=rs[ok])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int)
    a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    rars = sorted(RAW_DIR.glob("vin*.rar"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))[:a.n]
    for k, p in enumerate(rars):
        o = OUT / f"{p.stem}.npz"
        if o.exists():
            continue
        t0 = time.time()
        try:
            np.savez(o, **one(p)); log(f"[{k + 1}/{len(rars)}] {p.stem} {time.time() - t0:.0f}s")
        except Exception as ex:
            log(f"[{k + 1}/{len(rars)}] {p.stem} FAILED {ex!r}")