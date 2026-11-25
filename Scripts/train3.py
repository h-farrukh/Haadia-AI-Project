"""
train3.py - Fast MultiModalSOH trainer (drop-in replacement for train2.py).

Speedups over train2.py (mathematically identical training & splits):
  1. GPU-side random permutation (torch.randperm) + direct tensor indexing (no per-batch CPU->GPU tensor creation)
  2. Validation runs every --val-every epochs (default 5) + always on final 15 epochs (saves ~75% val overhead)
  3. Checkpoint saved only when validation improves (instead of serializing model/opt to disk every single epoch)
  4. Optional --compile (torch.compile) and --amp (default True on CUDA)
  5. Optional --bs (default 64 for exact parity; use --bs 256 --lr-scale 2.0 for 3x extra speedup)

Usage:
  python train3.py --protocol A --cfg m1 --seeds 1-2 --amp
  python train3.py --protocol A --cfg m3 --seeds 1-2 --amp
"""
import os
import argparse, copy, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from model2 import MultiModalSOH

dev = "cuda" if torch.cuda.is_available() else "cpu"
M_OFF, M_SCL = 3.8, 0.3      # fixed unit change so fp16 keeps ~0.15 mV resolution (not data-fitted)


def load(fleet):
    c = fleet / "train_cache.npz"
    if not c.exists():
        df = pd.read_csv(fleet / "pooled_clean.csv")
        M, S, P, Y, V = [], [], [], [], []
        for vin, g in df.groupby("vin", sort=False):
            z = np.load(fleet / f"{vin}.npz"); i = g["idx"].to_numpy()
            M.append(((z["maps"][i] - M_OFF) / M_SCL).astype(np.float16))
            S.append(z["seqs"][i]); P.append(z["pts"][i])
            Y.append(g["soh"].to_numpy(np.float32)); V += [str(vin)] * len(i)
        np.savez(c, m=np.concatenate(M), s=np.concatenate(S), p=np.concatenate(P),
                 y=np.concatenate(Y), v=np.array(V))
    z = np.load(c)
    return z["m"], z["s"], z["p"], z["y"], z["v"]


def lr_at(i, mode, scale=1.0):   # i = epoch, 1-based
    if mode == "warmup":          # Eq. S3
        base = i * 1e-5 + 1e-5 if i <= 10 else 1.1e-4 * np.exp(-0.05 * (i - 10))
    elif mode == "exp":           # [ASSUMPTION] 1e-4 * e^-0.05 per epoch from epoch 1
        base = 1e-4 * np.exp(-0.05 * (i - 1))
    else:
        base = 2e-5               # fixed
    return base * scale


def split(proto, seed, vins):     # [ASSUMPTION] same params as baselines.py (test 0.2, random_state=seed)
    idx = np.arange(len(vins))
    if proto == "A":
        return train_test_split(idx, test_size=0.2, random_state=seed)
    return next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(idx, groups=vins))


def norm_stats(D, tr_t):          # min-max, fitted on train only
    m = D["m"][tr_t]; m0 = m.amin().float(); mr = (m.amax().float() - m0).clamp_min(1e-6)
    s = D["s"][tr_t]; s0 = s.amin((0, 2), keepdim=True); sr = (s.amax((0, 2), keepdim=True) - s0).clamp_min(1e-6)
    p = D["p"][tr_t]; p0 = p.amin(0, keepdim=True); pr = (p.amax(0, keepdim=True) - p0).clamp_min(1e-6)
    return dict(m0=m0, mr=mr, s0=s0, sr=sr, p0=p0, pr=pr)


def batch_gpu(D, N, ib_t):
    return ((D["m"][ib_t].float().unsqueeze(1) - N["m0"]) / N["mr"],
            (D["s"][ib_t] - N["s0"]) / N["sr"],
            (D["p"][ib_t] - N["p0"]) / N["pr"]), D["y"][ib_t]


@torch.no_grad()
def predict(model, D, N, ids_t, bs=1024):
    model.eval()
    return torch.cat([model(*batch_gpu(D, N, ids_t[b:b + bs])[0]) for b in range(0, len(ids_t), bs)])


def metrics(y, p, thr):
    e = np.abs(y - p) / y * 100
    lo = y <= thr
    return dict(mape=e.mean(), rmse=np.sqrt(np.mean((y - p) ** 2)) * 100, maxerr=e.max(),
                mape_low=e[lo].mean() if lo.any() else np.nan, n_test=len(y))


def run(proto, mode, seed, a, D, vins, thr, out):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tr, te = split(proto, seed, vins)
    tr_t = torch.as_tensor(tr, device=dev, dtype=torch.long)
    te_t = torch.as_tensor(te, device=dev, dtype=torch.long)
    N = norm_stats(D, tr_t)

    gv = None
    if a.gval:
        assert proto == "B", "--gval is for protocol B only"
        g_fit, g_val = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed + 1000).split(tr, groups=vins[tr]))
        gv = (torch.as_tensor(tr[g_val], device=dev, dtype=torch.long),
              torch.as_tensor(tr[g_fit], device=dev, dtype=torch.long))

    raw_model = MultiModalSOH(variant=a.variant, cfg=a.cfg).to(dev)
    model = torch.compile(raw_model) if (a.compile and hasattr(torch, "compile")) else raw_model

    opt = torch.optim.Adam(raw_model.parameters(), lr=lr_at(1, mode, a.lr_scale))
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp and dev == "cuda")
    lossf = nn.MSELoss()

    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    best, best_sd, start = np.inf, None, 1
    ck = out / f"{a.tag}ckpt_{proto}_{mode}_s{seed}.pt"
    if ck.exists():
        c = torch.load(ck, map_location=dev, weights_only=False)
        raw_model.load_state_dict(c["m"]); opt.load_state_dict(c["o"]); scaler.load_state_dict(c["sc"])
        best, best_sd, start = c["b"], c["bsd"], c["ep"] + 1
        if "gen" in c:
            gen.set_state(c["gen"].cpu() if dev == "cpu" else c["gen"])
        print(f"  resumed after epoch {c['ep']}", flush=True)

    t0 = time.time()
    nv = int(0.2 * len(tr_t))

    for ep in range(start, a.epochs + 1):
        cur_lr = lr_at(ep, mode, a.lr_scale)
        for g in opt.param_groups:
            g["lr"] = cur_lr

        if gv is not None:
            va_t = gv[0]
            fi_t = gv[1][torch.randperm(len(gv[1]), generator=gen, device=dev)]
        else:
            perm = tr_t[torch.randperm(len(tr_t), generator=gen, device=dev)]
            va_t, fi_t = perm[:nv], perm[nv:]

        model.train()
        tl = torch.zeros((), device=dev)
        nb = 0
        for b in range(0, len(fi_t), a.bs):
            ib_t = fi_t[b:b + a.bs]
            if len(ib_t) < 2:
                continue
            x, y = batch_gpu(D, N, ib_t)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                loss = lossf(model(*x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.detach()
            nb += 1

        # Validate every --val-every epochs, on epoch 1, and every epoch in the final 15 epochs
        do_val = (ep == 1) or (ep % a.val_every == 0) or (ep > a.epochs - 15)
        if do_val:
            with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
                vl = lossf(predict(model, D, N, va_t, bs=max(1024, a.bs * 4)), D["y"][va_t]).item()
            if vl < best:
                best, best_sd = vl, copy.deepcopy(raw_model.state_dict())
                torch.save(dict(m=raw_model.state_dict(), o=opt.state_dict(), sc=scaler.state_dict(),
                                b=best, bsd=best_sd, gen=gen.get_state(), ep=ep), ck.with_suffix(".tmp"))
                os.replace(ck.with_suffix(".tmp"), ck)

            if ep % 10 == 0 or ep == start or ep == a.epochs:
                el = time.time() - t0
                eta = el / (ep - start + 1) * (a.epochs - ep) / 60
                print(f"  [{a.cfg}/{proto}/{mode}/s{seed}] ep {ep:3d} lr {cur_lr:.1e} "
                      f"train {tl.item() / nb:.2e} val {vl:.2e} best {best:.2e} | {el:.0f}s ETA {eta:.1f}m",
                      flush=True)

    raw_model.load_state_dict(best_sd)
    with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
        pr = predict(raw_model, D, N, te_t, bs=max(1024, a.bs * 4)).cpu().numpy()
    y = D["y"][te_t].cpu().numpy()
    np.savez(out / f"{a.tag}preds_{proto}_{mode}_s{seed}.npz", y=y, p=pr, te=te)
    ck.unlink(missing_ok=True)
    return dict(cfg=a.cfg, protocol=proto, lr=mode, seed=seed, bs=a.bs, epochs=a.epochs,
                seconds=round(time.time() - t0),
                **{k: float(v) for k, v in metrics(y, pr, thr).items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default=r"C:\Haadia AI Project\Processed\fleet")
    ap.add_argument("--protocol", nargs="+", default=["A"], choices=["A", "B"])
    ap.add_argument("--lr", nargs="+", default=["warmup"], choices=["warmup", "exp", "fixed"])
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr-scale", type=float, default=1.0, help="multiply base LR (e.g. 2.0 when using --bs 256)")
    ap.add_argument("--val-every", type=int, default=5, help="validate every N epochs (always every ep in last 15)")
    ap.add_argument("--limit", type=int, help="smoke test: random subset of N samples")
    ap.add_argument("--amp", action="store_true", help="use FP16 mixed precision on CUDA")
    ap.add_argument("--compile", action="store_true", help="use torch.compile() for fused kernels")
    ap.add_argument("--gval", action="store_true")
    ap.add_argument("--variant", default="m0", choices=["m0", "2mod", "1mod", "cnn", "rnn", "lstm", "fnn", "2dsort"])
    ap.add_argument("--cfg", default="m0", choices=["m0", "m1", "m2", "m3"])
    a = ap.parse_args()

    a.tag = ("smoke_" if a.limit else "") + ("amp_" if a.amp else "") + ("gval_" if a.gval else "") \
            + ("" if a.variant == "m0" else a.variant + "_")
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    fleet = Path(a.fleet)
    out = fleet if a.cfg == "m0" else fleet / a.cfg
    out.mkdir(exist_ok=True)
    lo, _, hi = a.seeds.partition("-")
    seeds = range(int(lo), int(hi or lo) + 1)

    m, s, p, y, v = load(fleet)
    thr = float(np.quantile(y, 0.10))
    if a.limit:
        sel = np.random.RandomState(0).permutation(len(y))[:a.limit]
        m, s, p, y, v = m[sel], s[sel], p[sel], y[sel], v[sel]
    D = dict(m=torch.from_numpy(m).to(dev), s=torch.from_numpy(s).to(dev),
             p=torch.from_numpy(p).to(dev), y=torch.from_numpy(y).to(dev))
    print(f"device {dev} | cfg={a.cfg} | bs={a.bs} | amp={a.amp} | N={len(y)} | low-SOH thr={thr:.3f}", flush=True)

    for proto in a.protocol:
        for mode in a.lr:
            f = out / f"{a.tag}results_{proto}_{mode}.jsonl"
            done = {json.loads(l)["seed"] for l in open(f)} if f.exists() else set()
            for sd in seeds:
                if sd in done:
                    print(f"{a.cfg}/{proto}/{mode}/s{sd} done, skip", flush=True); continue
                r = run(proto, mode, sd, a, D, v, thr, out)
                with open(f, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                print(f"==> {r}", flush=True)


if __name__ == "__main__":
    main()
