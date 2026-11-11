import numpy as np
import torch

from config import CHANNELS
from data import REGION

EARLY = 0.002


@torch.no_grad()
def predict(model, case, chunk=40, mode="direct"):
    model.eval()
    T = case["fields"].shape[0]
    if mode == "ar":
        x = case["fields"][0:1]
        cb = case["coords"][None]; sb = case["static"][None]; nb = case["cond"][None]
        preds = []
        for k in range(1, T):
            x = model(x, cb, sb, nb, case["tau"][k:k + 1]); preds.append(x)
        return torch.cat(preds, 0)
    preds = []
    wv = "wv_skel" in case
    wv_cm = "wv_coords" in case
    x0 = None if wv else case["fields"][0]
    for s0 in range(1, T, chunk):
        js = torch.arange(s0, min(s0 + chunk, T))
        B = len(js)
        if wv:
            xb = case["wv_skel"][js]
            cb = case["wv_coords"][js] if wv_cm else case["coords"][None].expand(B, -1, -1)
        else:
            xb = x0[None].expand(B, -1, -1)
            cb = case["coords"][None].expand(B, -1, -1)
        sb = case["static"][None].expand(B, -1, -1)
        nb = case["cond"][None].expand(B, -1, -1)
        preds.append(model(xb, cb, sb, nb, case["tau"][js]))
    return torch.cat(preds, 0)


@torch.no_grad()
def region_channel_rel(model, cases, nz, keys, mode="direct"):
    per = {c: [] for c in CHANNELS}
    allrel = []
    for k in keys:
        c = cases[k]
        pr = nz.denormalize(predict(model, c, mode=mode).cpu()).numpy()
        tr = c["fphys"][1:].cpu().numpy()
        bid = c["block_id"].numpy()
        for bv in REGION:
            m = bid == bv
            rel = np.sqrt(((pr[:, m] - tr[:, m]) ** 2).sum((0, 1))) / (np.sqrt((tr[:, m] ** 2).sum((0, 1))) + 1e-9)
            allrel.append(rel)
            for i, ch in enumerate(CHANNELS):
                per[ch].append(rel[i])
    balanced = float(np.stack(allrel).mean())
    chan = {c: float(np.mean(per[c])) for c in CHANNELS}
    return balanced, chan


@torch.no_grad()
def shock_metrics(model, cases, nz, keys, mode="direct"):
    pv, epv = [], []
    for k in keys:
        c = cases[k]
        pr = nz.denormalize(predict(model, c, mode=mode).cpu()).numpy()
        tr = c["fphys"][1:].cpu().numpy()
        pm = c["block_id"].numpy() == 2
        tau = c["tau"][1:].cpu().numpy()
        p = pr[:, pm, 0]
        t = tr[:, pm, 0]
        pv.append(np.sqrt(((p - t) ** 2).sum()) / (np.sqrt((t ** 2).sum()) + 1e-9))
        em = tau < EARLY
        pe, te = p[em], t[em]
        epv.append(np.sqrt(((pe - te) ** 2).sum()) / (np.sqrt((te ** 2).sum()) + 1e-9))
    return float(np.mean(pv)), float(np.mean(epv))

