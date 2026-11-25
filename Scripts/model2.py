import torch
import torch.nn as nn
import torch.nn.functional as F


class SameConv(nn.Module):
    """Conv with TF-style 'same' padding: output length = ceil(L / stride), any kernel size."""
    def __init__(self, dim, cin, cout, k, stride=1):
        super().__init__()
        self.k, self.s = k, stride
        self.conv = (nn.Conv2d if dim == 2 else nn.Conv1d)(cin, cout, k, stride)

    def forward(self, x):
        pads = []
        for L in reversed(x.shape[2:]):                 # F.pad wants the last dimension first
            out = -(-L // self.s)
            tot = max((out - 1) * self.s + self.k - L, 0)
            pads += [tot // 2, tot - tot // 2]
        return self.conv(F.pad(x, pads))


class BasicBlock(nn.Module):
    def __init__(self, dim, cin, cout, k, stride, shortcut=True):
        super().__init__()
        bn = nn.BatchNorm2d if dim == 2 else nn.BatchNorm1d
        self.f = nn.Sequential(SameConv(dim, cin, cout, k, stride), bn(cout), nn.ReLU(),
                               SameConv(dim, cout, cout, k, 1), bn(cout))
        self.shortcut = shortcut
        if shortcut:
            self.sc = nn.Identity() if (cin == cout and stride == 1) else \
                nn.Sequential(SameConv(dim, cin, cout, 1, stride), bn(cout))

    def forward(self, x):
        y = self.f(x)
        return torch.relu(y + self.sc(x)) if self.shortcut else torch.relu(y)


class ResBranch(nn.Module):
    """input conv -> BN -> ReLU -> maxpool -> 4 residual blocks (2 basic blocks each) -> GAP -> 3 FC."""
    def __init__(self, dim, cin, filters, k, out=64, shortcut=True):
        super().__init__()
        assert len(filters) == 8
        bn, pool, gap = (nn.BatchNorm2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d) if dim == 2 else \
                        (nn.BatchNorm1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d)
        self.stem = nn.Sequential(SameConv(dim, cin, filters[0], k), bn(filters[0]), nn.ReLU(), pool(2))
        blocks, c = [], filters[0]
        for i, f in enumerate(filters):                  # BB1..BB8; RB2-RB4 start with stride 2
            blocks.append(BasicBlock(dim, c, f, k, 2 if i in (2, 4, 6) else 1, shortcut))
            c = f
        self.res = nn.Sequential(*blocks)
        self.gap = gap(1)
        self.fc = nn.Sequential(nn.Linear(c, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                nn.Linear(64, out), nn.ReLU())

    def forward(self, x):
        return self.fc(self.gap(self.res(self.stem(x))).flatten(1))


# Supp. Table 4, model M0
F2D = (16, 16, 32, 32, 32, 32, 64, 64)
F1D = (16, 16, 32, 32, 64, 64, 128, 128)
CFGS = {"m0": (F2D, F1D),                                  # Supp. Table 4 (m1-m3 are symmetric 2D/1D)
        "m1": ((8, 8, 16, 16, 16, 16, 32, 32),) * 2,
        "m2": ((8, 8, 16, 16, 32, 32, 64, 64),) * 2,
        "m3": ((16, 16, 32, 32, 64, 64, 128, 128),) * 2}


class SeqRec(nn.Module):
    """RNN/LSTM replacement for the 1D branch: (N,2,151) -> (N,151,2) -> last step -> 3 FC."""
    def __init__(self, kind, hid=64, out=64):          # [ASSUMPTION] 1 layer, hidden 64
        super().__init__()
        self.r = (nn.RNN if kind == "rnn" else nn.LSTM)(2, hid, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hid, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                nn.Linear(64, out), nn.ReLU())

    def forward(self, x):
        return self.fc(self.r(x.transpose(1, 2))[0][:, -1])


def seq_branch(kind, sc, f1d=F1D):
    if kind in ("rnn", "lstm"):
        return SeqRec(kind)
    if kind == "fnn":                                   # [ASSUMPTION] 302->128->64->64
        return nn.Sequential(nn.Flatten(), nn.Linear(302, 128), nn.ReLU(), nn.Linear(128, 64),
                             nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
    return ResBranch(1, 2, f1d, k=30, shortcut=sc)


VARIANTS = ("m0", "2mod", "1mod", "cnn", "rnn", "lstm", "fnn", "2dsort")


class MultiModalSOH(nn.Module):
    """m0 = full; 2mod = no 2D map; 1mod = points only; cnn = no shortcuts;
    rnn/lstm/fnn = 1D ResNet branch replaced (2D branch kept)."""
    def __init__(self, n_points=15, variant="m0", cfg="m0"):
        super().__init__()
        assert variant in VARIANTS
        f2d, f1d = CFGS[cfg]
        sc = variant != "cnn"
        self.sort2d = variant == "2dsort"
        self.use2d = variant not in ("2mod", "1mod")
        self.use1d = variant != "1mod"
        if self.use2d:
            self.b2d = ResBranch(2, 1, f2d, k=5, shortcut=sc)
        if self.use1d:
            self.b1d = seq_branch(variant, sc, f1d)
        self.bpt = nn.Sequential(nn.Linear(n_points, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(64 * (1 + self.use2d + self.use1d), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, vmap, seq, pts):                   # (N,1,96,96), (N,2,151), (N,15)
        z = []
        if self.use2d: z.append(self.b2d(vmap.sort(dim=3).values if self.sort2d else vmap))
        if self.use1d: z.append(self.b1d(seq))
        z.append(self.bpt(pts))
        return self.head(torch.cat(z, 1)).squeeze(1)


if __name__ == "__main__":
    for c in CFGS:
        m = MultiModalSOH(cfg=c).eval()
        print(f"cfg {c}: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M")
    for v in VARIANTS:
        m = MultiModalSOH(variant=v).eval()
        out = m(torch.randn(2, 1, 96, 96), torch.randn(2, 2, 151), torch.randn(2, 15))
        print(f"{v:5s} params {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M | out {tuple(out.shape)}")


