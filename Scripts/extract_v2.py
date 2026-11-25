"""
extract_v2.py - like extract.py, but keeps rest/anchor fields so SOC can be re-derived from an OCV curve.
Self-contained (no import from extract.py). Writes to Processed\\ocv_pilot by default; never touches fleet.

  python extract_v2.py --n 10        # pilot: same 10 vehicles as extract.py --n 10 (seed 0)
  python extract_v2.py --only vin1
  python extract_v2.py --all         # resumable

Differences vs extract.py:
- no 10<soc_start<=50 filter; only soc_start <= MAX_START_SOC (paper's 0.6 skip)   [ASSUMPTION: BMS soc as loose pre-filter]
- extra per-segment fields: rest_s, pre_rest_s, v_start, i_start, v_prev, i_prev, tmin_start, ah, anchor
- anchor = pre_rest_s >= ANCH_REST_S and |i_prev| < ANCH_I_A; the OCV voltage is then v_prev.  [ASSUMPTION: thresholds]
"""
import argparse, csv, gc, json, os, time, traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Haadia AI Project")
RAW_DIR = ROOT / "Raw Data"
OUT_DIR = ROOT / "Processed" / "ocv_pilot"
UNRAR = r"C:\Program Files\WinRAR\UnRAR.exe"

RATED_AH = 155.0
REST_GAP_S = 3600
MAX_FRAME_GAP_S = 60
MAX_START_SOC = 60.0         # paper: skip if SOC_s > 0.6
MIN_VMAX_END = 4.22
MIN_CHARGE_FRAC = 0.9
V_HI = np.arange(3900, 4051) / 1000.0
CAND_V = (3.85, 4.06)
N_CELLS, N_PROBES = 96, 32
MIN_GOOD_FRAMES, MIN_GOOD_FRAC = 20, 0.8
OUTLIER_PCT = 8.0
ANCH_REST_S, ANCH_I_A = 3600, 2.0
CHUNK = 200_000

NUM = ["terminaltime", "soc", "totalodometer", "chargestatus", "totalcurrent",
       "minvoltagebattery", "maxvoltagebattery", "mintemperaturevalue"]
STR = ["batteryvoltage", "probetemperatures"]
T_, SOC_, ODO_, CS_, CUR_, VMIN_, VMAX_, TMIN_ = range(8)
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


@contextmanager
def open_csv(path):
    path = Path(path)
    if path.suffix.lower() == ".rar":
        import rarfile
        rarfile.UNRAR_TOOL = UNRAR
        with rarfile.RarFile(path) as a:
            name = [f for f in a.namelist() if f.lower().endswith(".csv")][0]
            with a.open(name) as f:
                yield f, a.getinfo(name).file_size / 1e6
    else:
        with open(path, "rb") as f:
            yield f, path.stat().st_size / 1e6


def split_arr(s, n):
    s = s.fillna("").astype(str)
    out = np.full((len(s), n), np.nan, dtype=np.float32)
    ok = (s.str.count("~") == n - 1) & ~s.str.contains("~~", regex=False) \
         & ~s.str.startswith("~") & ~s.str.endswith("~")
    good = s[ok]
    if len(good) == 0:
        return out
    try:
        flat = np.fromstring("~".join(good.tolist()), sep="~", dtype=np.float32)
        if flat.size != len(good) * n:
            raise ValueError("size mismatch")
        out[ok.values] = flat.reshape(-1, n)
    except ValueError:
        for i, row in zip(np.flatnonzero(ok.values), good.tolist()):
            try:
                out[i] = np.array(row.split("~"), dtype=np.float32)
            except ValueError:
                pass
    return out


class Cand:
    def __init__(self, pos, cells, temps):
        o = np.argsort(pos)
        self.pos, self.cells, self.temps = pos[o], cells[o], temps[o]

    def get(self, a, b):
        idx = np.arange(a, b + 1)
        c = np.full((len(idx), N_CELLS), np.nan, np.float32)
        t = np.full((len(idx), N_PROBES), np.nan, np.float32)
        if len(self.pos):
            k = np.minimum(np.searchsorted(self.pos, idx), len(self.pos) - 1)
            hit = self.pos[k] == idx
            c[hit], t[hit] = self.cells[k[hit]], self.temps[k[hit]]
        return c, t


def read_vehicle(path):
    num_parts, cpos, ccells, ctemps = [], [], [], []
    n = 0
    est_total = t_first = None
    with open_csv(path) as (f, size_mb):
        log(f"  {Path(path).name}: {size_mb:.0f} MB uncompressed")
        for ch in pd.read_csv(f, encoding="utf-8-sig", chunksize=CHUNK, usecols=NUM + STR,
                              dtype={c: str for c in STR}):
            for c in NUM:
                if ch[c].dtype == object:
                    ch[c] = pd.to_numeric(ch[c], errors="coerce")
            x = ch[NUM].to_numpy(dtype=np.float64)
            cand = (x[:, CUR_] < 0) & (x[:, VMAX_] >= CAND_V[0]) & (x[:, VMAX_] <= CAND_V[1])
            if cand.any():
                sub = ch[cand]
                ccells.append(split_arr(sub["batteryvoltage"], N_CELLS))
                ctemps.append(split_arr(sub["probetemperatures"], N_PROBES))
                cpos.append(n + np.flatnonzero(cand))
            num_parts.append(x)
            if est_total is None:
                bpr = ch["batteryvoltage"].astype(str).str.len().mean() \
                      + ch["probetemperatures"].astype(str).str.len().mean() + 60
                est_total, t_first = size_mb * 1e6 / bpr, time.time()
            n += len(ch)
            done = min(n / est_total, 0.999)
            eta = (time.time() - t_first) / max(done, 1e-9) * (1 - done) / 60
            log(f"  read {n:,} rows | ~{100 * done:.0f}% | ETA ~{eta:.1f} min")
    X = np.vstack(num_parts); del num_parts
    order = np.argsort(X[:, T_], kind="stable")
    inv = np.empty(n, np.int64); inv[order] = np.arange(n)
    Xs = X[order]; del X
    ts = Xs[:, T_]
    keep = ~np.isnan(ts)
    keep[1:] &= ts[1:] != ts[:-1]
    final_idx = np.cumsum(keep) - 1
    final_idx[~keep] = -1
    A = Xs[keep]; del Xs
    if cpos:
        cp = final_idx[inv[np.concatenate(cpos)]]
        ok = cp >= 0
        cand = Cand(cp[ok], np.vstack(ccells)[ok], np.vstack(ctemps)[ok])
    else:
        cand = Cand(np.zeros(0, np.int64), np.zeros((0, N_CELLS), np.float32), np.zeros((0, N_PROBES), np.float32))
    return A, cand, dict(size_mb=size_mb, rows=n, dup_or_nan_rows=int(n - keep.sum()), cand_rows=len(cand.pos))


def find_segments(A, skip):
    t, soc, cur = A[:, T_], A[:, SOC_], A[:, CUR_]
    vmax, vmin, cs, odo, tmin = A[:, VMAX_], A[:, VMIN_], A[:, CS_], A[:, ODO_], A[:, TMIN_]
    bad = np.isnan(cur) | np.isnan(vmax) | np.isnan(vmin) | np.isnan(soc) \
          | (cur < -400) | (cur > 600) | (vmin < 2.0) | (cs == 255)
    bad_cum = np.concatenate([[0], np.cumsum(bad)])
    dt = np.diff(t, prepend=t[0])
    act = np.isnan(cur) | (np.abs(cur) >= 2.0)
    last_act = np.maximum.accumulate(np.where(act, np.arange(len(t)), -1))
    gap_idx = np.flatnonzero(dt > REST_GAP_S)
    idx100 = np.flatnonzero(soc >= 100)
    charge_idx = np.flatnonzero((cs == 1) & (np.r_[0, cs[:-1]] != 1))
    skip["charge_runs"] = len(charge_idx)
    best = {}
    for s in charge_idx:
        if not (soc[s] <= MAX_START_SOC):
            skip["start_soc_too_high"] = skip.get("start_soc_too_high", 0) + 1
            continue
        g = np.searchsorted(gap_idx, s, side="right")
        nxt = gap_idx[g] if g < len(gap_idx) else len(t)
        j = np.searchsorted(idx100, s)
        if j >= len(idx100) or idx100[j] >= nxt:
            skip["no_full_charge"] = skip.get("no_full_charge", 0) + 1
            continue
        e = idx100[j]
        if e <= s + 1 or dt[s + 1:e + 1].max() > MAX_FRAME_GAP_S:
            skip["frame_gaps"] = skip.get("frame_gaps", 0) + 1
            continue
        if bad_cum[e + 1] - bad_cum[s] > 0:
            skip["invalid_frames"] = skip.get("invalid_frames", 0) + 1
            continue
        if vmax[e] < MIN_VMAX_END:
            skip["vmax_end_low"] = skip.get("vmax_end_low", 0) + 1
            continue
        i = cur[s:e + 1]
        if (i < 2.0).mean() < MIN_CHARGE_FRAC:
            skip["not_pure_charge"] = skip.get("not_pure_charge", 0) + 1
            continue
        ah = -np.sum(0.5 * (i[1:] + i[:-1]) * np.diff(t[s:e + 1])) / 3600.0
        cap = ah / ((100.0 - soc[s]) / 100.0)
        if ah <= 0 or not (0.3 * RATED_AH < cap < 1.5 * RATED_AH):
            skip["bad_capacity"] = skip.get("bad_capacity", 0) + 1
            continue
        if e not in best or soc[s] < best[e]["soc_start"]:
            p = max(s - 1, 0)
            pre_rest = 0.0 if act[p] else float(t[p] - t[max(last_act[p], 0)])
            best[e] = dict(s=int(s), e=int(e), soc_start=float(soc[s]), vmax_end=float(vmax[e]),
                           capacity=float(cap), soh=float(cap / RATED_AH), day=float(t[s] / 86400),
                           odo_km=float(odo[s]), ah=float(ah),
                           rest_s=float(dt[s]), pre_rest_s=pre_rest,
                           v_start=float(vmax[s]), i_start=float(cur[s]),
                           v_prev=float(vmax[p]), i_prev=float(cur[p]), tmin_start=float(tmin[s]))
    return sorted(best.values(), key=lambda d: d["s"])


def interp_rows(a, n_out):
    pos = np.linspace(0, len(a) - 1, n_out)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(a) - 1)
    f = pos - lo
    if a.ndim == 2:
        f = f[:, None]
    return a[lo] * (1 - f) + a[hi] * f


def extract_hi(sg, A, cand):
    s, e = sg["s"], sg["e"]
    t, cur, vm = A[s:e + 1, T_], -A[s:e + 1, CUR_], A[s:e + 1, VMAX_]
    above = np.flatnonzero(vm >= V_HI[0])
    if len(above) == 0:
        return None, "never_reaches_3900mV"
    i1 = above[0]
    top = np.flatnonzero(vm[i1:] >= V_HI[-1])
    if len(top) == 0:
        return None, "never_reaches_4050mV"
    i2 = i1 + top[0]
    if vm[i1] > V_HI[0] + 0.01:
        return None, "starts_above_window"
    w = slice(i1, i2 + 1)
    if np.diff(t[w]).max() > MAX_FRAME_GAP_S:
        return None, "frame_gap_in_window"
    q = np.concatenate([[0], np.cumsum(0.5 * (cur[w][1:] + cur[w][:-1]) * np.diff(t[w]) / 3600.0)])
    q_hi = np.interp(V_HI, np.maximum.accumulate(vm[w]), q)
    c, tp = cand.get(s + i1, s + i2)
    ok = ~np.isnan(c).any(1) & ~np.isnan(tp).any(1)
    n_win = i2 - i1 + 1
    if ok.sum() < MIN_GOOD_FRAMES or ok.sum() < MIN_GOOD_FRAC * n_win:
        return None, "too_few_good_frames"
    cw, tw = c[ok], tp[ok]
    if cw.min() < 2.0:
        return None, "implausible_cell_voltage"
    vmap = interp_rows(cw, 96)
    tavg = tw.mean(1)
    t_hi = interp_rows(tavg, 151)
    dv = (cw.max(1) - cw.min(1)) * 1000.0
    tmin = A[s:e + 1, TMIN_]
    c_chg = cur[cur > 0]
    pts = [A[s, ODO_], tavg.mean(), q_hi.mean(), np.median(q_hi), q_hi.std(), np.ptp(q_hi),
           dv.mean(), np.median(dv), dv.std(), np.ptp(dv),
           c_chg.max(), c_chg.mean(), sg["vmax_end"], np.nanmin(tmin), np.nanmean(tmin)]
    return (vmap.astype(np.float32), np.stack([q_hi, t_hi]).astype(np.float32),
            np.array(pts, dtype=np.float32)), "ok"


EXTRA = ["ah", "rest_s", "pre_rest_s", "v_start", "i_start", "v_prev", "i_prev", "tmin_start"]


def process_vehicle(path, out_dir):
    vin = Path(path).stem
    t_start = time.time()
    A, cand, info = read_vehicle(path)
    skip = {}
    segs = find_segments(A, skip)
    log(f"  {len(segs)} label segments | charge runs {skip.get('charge_runs', 0)} | skipped {skip}")
    maps, seqs, pts, keep = [], [], [], []
    for k, sg in enumerate(segs):
        res, why = extract_hi(sg, A, cand)
        if res is None:
            skip["hi_" + why] = skip.get("hi_" + why, 0) + 1
            continue
        maps.append(res[0]); seqs.append(res[1]); pts.append(res[2]); keep.append(k)
    S = pd.DataFrame([segs[k] for k in keep])
    n = len(S)
    n_anchor = 0
    if n:
        soh = S["soh"]
        trend = soh.rolling(31, center=True, min_periods=5).median()
        outlier = ((soh - trend).abs() / trend * 100 > OUTLIER_PCT).fillna(False).astype(np.int8).values
        anchor = ((S["pre_rest_s"] >= ANCH_REST_S) & (S["i_prev"].abs() < ANCH_I_A)).values.astype(np.int8)
        n_anchor = int(anchor.sum())
        extra = {k: S[k].values.astype(np.float32) for k in EXTRA}
        tmp = Path(out_dir) / f"{vin}.tmp.npz"
        np.savez_compressed(tmp, maps=np.array(maps), seqs=np.array(seqs), pts=np.array(pts),
                            soh=soh.values.astype(np.float32), day=S["day"].values, odo=S["odo_km"].values,
                            soc_start=S["soc_start"].values, vmax_end=S["vmax_end"].values,
                            outlier=outlier, anchor=anchor, vin=np.array(vin), **extra)
        os.replace(tmp, Path(out_dir) / f"{vin}.npz")
    else:
        outlier = np.zeros(0)
    info.update(vin=vin, segments=len(segs), samples=n, outliers=int(outlier.sum()), anchors=n_anchor,
                seconds=round(time.time() - t_start, 1), skip=json.dumps(skip), status="ok")
    log(f"  -> {vin}: {n} samples ({n_anchor} anchors, {int(outlier.sum())} outliers) in {info['seconds']}s")
    return info


def ordered_rars(seed):
    rars = sorted(RAW_DIR.glob("vin*.rar"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    perm = np.random.RandomState(seed).permutation(len(rars))
    return [rars[i] for i in perm]


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="one vehicle, e.g. vin1")
    ap.add_argument("--n", type=int, help="first N vehicles of the fixed random order")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--file", help="test: path to a plain .csv instead of a RAR")
    ap.add_argument("--out", help="output folder override")
    a = ap.parse_args()
    if a.out:
        OUT_DIR = Path(a.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if a.file:
        todo = [Path(a.file)]
    elif a.only:
        todo = [RAW_DIR / f"{a.only}.rar"]
    elif a.n or a.all:
        todo = ordered_rars(a.seed)[: (a.n if a.n else None)]
    else:
        ap.error("give --only, --n, --all or --file")

    man = OUT_DIR / "manifest.csv"
    cols = ["vin", "status", "size_mb", "rows", "dup_or_nan_rows", "cand_rows", "segments", "samples",
            "outliers", "anchors", "seconds", "skip", "error"]
    if not man.exists():
        with open(man, "w", newline="") as f:
            csv.DictWriter(f, cols).writeheader()
    done = 0
    for k, p in enumerate(todo):
        vin = p.stem
        if (OUT_DIR / f"{vin}.npz").exists():
            log(f"[{k + 1}/{len(todo)}] {vin}: already done, skipping")
            continue
        log(f"[{k + 1}/{len(todo)}] {vin}")
        try:
            info = process_vehicle(p, OUT_DIR)
        except Exception as ex:
            traceback.print_exc()
            info = dict(vin=vin, status="FAILED", error=repr(ex)[:200])
        with open(man, "a", newline="") as f:
            csv.DictWriter(f, cols, extrasaction="ignore", restval="").writerow(info)
        done += 1
        gc.collect()
    log(f"finished {done} vehicle(s); results in {OUT_DIR}")


if __name__ == "__main__":
    main()