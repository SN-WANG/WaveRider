import argparse
import os

import numpy as np
import torch

from config import CKPT_DIR, MODEL_CONFIGS, N_FOLDS, WV_ALIGN
from data import get_cv_folds, load_cases
from infer import region_channel_rel, shock_metrics
from model import build_model
from normalization import Normalizer


def _load(path, cfg, dev):
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}\nTrain it first: `python train.py --config <cfg> --fold <k>`.")
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build_model(cfg, ck["hp"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    nzsd = ck["normalizer"]
    nz = Normalizer(nzsd["mean"], nzsd["std"], nzsd.get("channels"))
    return m, nz, ck["test_keys"]


def eval_config(name, dev):
    cfg = MODEL_CONFIGS[name]
    mode = cfg.get("mode", "direct")
    rows = []
    for k in range(N_FOLDS):
        m, nz, test_keys = _load(f"{CKPT_DIR}/{name}/fold{k}/last.pt", cfg, dev)
        assert sorted(test_keys) == sorted(get_cv_folds()[k][1]), f"{name} fold{k} test-key mismatch"
        cases = load_cases(test_keys, nz, dev, waverider=bool(cfg.get("waverider")),
                           wv_align=WV_ALIGN, wv_fixed=bool(cfg.get("wv_fixed")))
        bal, ch = region_channel_rel(m, cases, nz, test_keys, mode=mode)
        pv, epv = shock_metrics(m, cases, nz, test_keys, mode=mode)
        rows.append(dict(bal=bal, pipe_vz=pv, early_vz=epv, **ch))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    ap.add_argument("--base", default="transolver")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    agg = {}
    print(f"{'config':10}{'CV-balanced':>14}{'v_z':>8}{'v_r':>8}{'p':>8}{'T':>8}{'rho':>8}{'pipeVz':>9}{'earlyVz':>9}   (mean+-std over 5 folds)")
    for name in a.configs:
        rows = eval_config(name, dev)
        agg[name] = rows
        mean = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
        std = {k: np.std([r[k] for r in rows]) for k in rows[0]}
        print(f"{name:10}{mean['bal']:.4f}+-{std['bal']:.4f}{mean['v_z']:8.3f}{mean['v_r']:8.3f}"
              f"{mean['p']:8.3f}{mean['T']:8.3f}{mean['rho']:8.3f}{mean['pipe_vz']:9.4f}{mean['early_vz']:9.4f}")

    if a.base in agg:
        crit = 2.776
        print(f"\nfold-paired vs {a.base} (|t|>{crit} significant, df={N_FOLDS - 1}):")
        for name in a.configs:
            if name == a.base:
                continue
            for key in ["bal", "pipe_vz", "early_vz"]:
                d = np.array([r[key] for r in agg[name]]) - np.array([r[key] for r in agg[a.base]])
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = d.mean() / se if se > 0 else 0.0
                print(f"  {name:10} {key:9} d={d.mean():+.4f} t={t:+.2f} ({(d < 0).sum()} better / {(d > 0).sum()} worse)")


if __name__ == "__main__":
    main()

