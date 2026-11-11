import csv
import math
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from config import CHANNELS, DATA_DIR, DOE_CSV, N_FOLDS, RGAS_N2, GAMMA, P_LOW
from normalization import Normalizer, phys_to_transformed
from riemann import sample_vec, solve_star

REGION = {0: "hp", 1: "lp", 2: "pipe"}
COORD_LO, COORD_HI = (-0.15, 0.0), (5.65, 0.65)


def _doe_path():
    return DOE_CSV if os.path.isabs(DOE_CSV) else os.path.join(os.path.dirname(__file__), DOE_CSV)


def all_keys():
    with open(_doe_path()) as fh:
        rows = list(csv.DictReader(fh))
    keys = [(float(r["dp_MPa"]), int(r["D_mm"])) for r in rows if r["set"] in ("train_pool", "test")]
    return sorted(keys)


def get_cv_folds():
    keys = all_keys()
    folds = []
    for k in range(N_FOLDS):
        test = keys[k::N_FOLDS]
        train = [x for x in keys if x not in test]
        folds.append((train, test))
    return folds


def _case_path(dp, D):
    return os.path.join(DATA_DIR, f"dp{float(dp)}_D{int(D)}.pt")


def bipolar(c):
    lo = torch.tensor(COORD_LO)
    hi = torch.tensor(COORD_HI)
    return (2 * (c - lo) / (hi - lo).clamp_min(1e-6) - 1).float()


def refit_normalizer(train_keys):
    C = len(CHANNELS)
    tot = torch.zeros(C, dtype=torch.float64)
    totsq = tot.clone()
    n = 0
    for dp, D in train_keys:
        f = torch.load(_case_path(dp, D), weights_only=False)["fields"].double()
        t = phys_to_transformed(f).reshape(-1, C)
        tot += t.sum(0)
        totsq += (t * t).sum(0)
        n += t.shape[0]
    mean = (tot / n).float()
    std = torch.sqrt((totsq / n - (tot / n) ** 2).clamp_min(1e-12)).float()
    return Normalizer(mean, std)


def build_waverider(case, d, nz, align=1.0):
    tphys = d["times"].numpy().astype(float)
    bid = case["block_id"].numpy()
    z = d["coords"][:, 0].numpy()
    N = case["N"]
    pipe = bid == 2
    zp = z[pipe]
    z0 = zp.min()
    L = zp.max() - z0
    T0 = float(case["fphys"][0, bid == 0, 3].mean())
    pL = float(case["fphys"][0, bid == 0, 2].mean())
    rL = pL / (RGAS_N2 * T0)
    rR = P_LOW / (RGAS_N2 * T0)
    ps, us, aL, aR = solve_star(pL, rL, 0.0, P_LOW, rR, 0.0, GAMMA)
    S_ana = aR * math.sqrt((GAMMA + 1) / (2 * GAMMA) * ps / P_LOW + (GAMMA - 1) / (2 * GAMMA))
    f0 = case["fphys"][0].cpu().numpy()
    base = bipolar(d["coords"]).numpy()
    Tn = len(tphys)
    skel = np.broadcast_to(f0, (Tn, N, 5)).copy()
    comov = np.broadcast_to(base, (Tn, N, 2)).copy()
    for k in range(Tn):
        t = max(tphys[k], 1e-12) * align
        rho, u, p = sample_vec((zp - z0) / t, pL, rL, 0.0, P_LOW, rR, 0.0, GAMMA)
        skel[k, pipe, 0] = u
        skel[k, pipe, 1] = 0.0
        skel[k, pipe, 2] = p
        skel[k, pipe, 3] = p / (rho * RGAS_N2)
        skel[k, pipe, 4] = rho
        zfront = z0 + min(S_ana * tphys[k], L)
        comov[k, pipe, 0] = np.clip((zp - zfront) / L, -1.0, 1.0)
    return nz.normalize(torch.from_numpy(skel).float()), torch.from_numpy(comov).float()


def load_cases(keys, nz, device, waverider=False, wv_align=1.0, wv_fixed=False):
    cases = {}
    for dp, D in keys:
        d = torch.load(_case_path(dp, D), weights_only=False)
        N = d["coords"].shape[0]
        bid = d["block_id"].long()
        f_phys = d["fields"].float()
        onehot = torch.zeros(N, 3).scatter_(1, bid.view(-1, 1), 1.0)
        area = d["nodal_area"].clamp_min(1e-12).log().view(-1, 1)
        area = (area - area.mean()) / area.std().clamp_min(1e-6)
        static = torch.cat([onehot, d["wall_mask"].float().view(-1, 1), d["axis_mask"].float().view(-1, 1),
                            d["iface_mask"].float().view(-1, 1), area], -1)
        blocksize = (torch.tensor([0.30, 1.30, D / 1000.0])[bid] / 1.30).view(-1)
        cond = torch.stack([torch.full((N,), math.log(dp)), torch.full((N,), (D - 10) / 20.0), blocksize], -1)
        case = dict(
            coords=bipolar(d["coords"]).to(device), static=static.to(device), cond=cond.to(device),
            fields=nz.normalize(f_phys).float().to(device), tau=d["tau_grid"].float().to(device),
            block_id=bid, N=N, fphys=f_phys.to(device))
        if waverider:
            skel, comov = build_waverider(case, d, nz, align=wv_align)
            case["wv_skel"] = skel.to(device)
            if not wv_fixed:
                case["wv_coords"] = comov.to(device)
        cases[(dp, D)] = case
    return cases


class DirectDS(Dataset):

    def __init__(self, cases, keys, stride=1, mode="direct"):
        hi = (lambda T: T - 1) if mode == "ar" else (lambda T: T)
        self.s = [(k, j) for k in keys for j in range(0, hi(cases[k]["fields"].shape[0]), stride)]

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        return self.s[i]


def collate(cases, mode="direct"):
    wv = "wv_skel" in next(iter(cases.values()))
    wv_cm = "wv_coords" in next(iter(cases.values()))
    if mode == "ar":
        def f(batch):
            keys = [b[0] for b in batch]; js = [b[1] for b in batch]
            return dict(
                x0=torch.stack([cases[k]["fields"][j] for k, j in zip(keys, js)]),
                tgt=torch.stack([cases[k]["fields"][j + 1] for k, j in zip(keys, js)]),
                tau=torch.stack([cases[k]["tau"][j + 1] for k, j in zip(keys, js)]),
                coords=torch.stack([cases[k]["coords"] for k in keys]),
                static=torch.stack([cases[k]["static"] for k in keys]),
                cond=torch.stack([cases[k]["cond"] for k in keys]))
        return f

    def f(batch):
        keys = [b[0] for b in batch]
        js = [b[1] for b in batch]
        out = dict(
            tgt=torch.stack([cases[k]["fields"][j] for k, j in zip(keys, js)]),
            tau=torch.stack([cases[k]["tau"][j] for k, j in zip(keys, js)]),
            static=torch.stack([cases[k]["static"] for k in keys]),
            cond=torch.stack([cases[k]["cond"] for k in keys]))
        if wv:
            out["x0"] = torch.stack([cases[k]["wv_skel"][j] for k, j in zip(keys, js)])
            if wv_cm:
                out["coords"] = torch.stack([cases[k]["wv_coords"][j] for k, j in zip(keys, js)])
            else:
                out["coords"] = torch.stack([cases[k]["coords"] for k in keys])
        else:
            out["x0"] = torch.stack([cases[k]["fields"][0] for k in keys])
            out["coords"] = torch.stack([cases[k]["coords"] for k in keys])
        return out
    return f


def wmse(pred, true, cw, eps=1e-6):
    err2 = (pred - true) ** 2
    denom = true.pow(2).mean(dim=tuple(range(true.dim() - 1)), keepdim=True).clamp_min(eps)
    return ((err2 / denom) * cw.view(*([1] * (true.dim() - 1)), -1)).mean()

