import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from config import CKPT_DIR, HP, MODEL_CONFIGS, N_FOLDS, WV_ALIGN
from data import DirectDS, collate, get_cv_folds, load_cases, refit_normalizer, wmse
from model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--fold", type=int, required=True, choices=list(range(N_FOLDS)))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=HP["epochs"])
    a = ap.parse_args()

    cfg = MODEL_CONFIGS[a.config]
    tag = f"{a.config}/fold{a.fold}"
    out = os.path.join(CKPT_DIR, tag)
    os.makedirs(out, exist_ok=True)
    log = open(os.path.join(out, "train.log"), "w")

    def P(*x):
        print(f"[{a.config} f{a.fold}]", *x)
        print(*x, file=log)
        log.flush()

    torch.manual_seed(a.seed)
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

    mode = cfg.get("mode", "direct")
    TRAIN, TEST = get_cv_folds()[a.fold]
    P(f"config={a.config} model={cfg.get('model','transolver')} mode={mode} fold={a.fold} "
      f"waverider={cfg.get('waverider')} wv_fixed={cfg.get('wv_fixed')} "
      f"head_reinject={cfg.get('head_reinject')} train={len(TRAIN)} test={len(TEST)}")

    nz = refit_normalizer(TRAIN)
    cases = load_cases(TRAIN, nz, dev, waverider=bool(cfg.get("waverider")),
                       wv_align=WV_ALIGN, wv_fixed=bool(cfg.get("wv_fixed")))

    c0 = cases[TRAIN[0]]
    err = float((nz.denormalize(c0["fields"].cpu()) - c0["fphys"].cpu()).abs().max()
                / (c0["fphys"].cpu().abs().max() + 1e-9))
    assert err < 1e-4, f"normalizer round-trip broken (rel_err={err:.2e})"
    P(f"normalizer round-trip rel_err={err:.2e}")

    cw = torch.tensor(HP["cw"]).to(dev)
    model = build_model(cfg, HP).to(dev)
    P(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    loader = DataLoader(DirectDS(cases, TRAIN, stride=1, mode=mode), batch_size=HP["batch_size"],
                        shuffle=True, collate_fn=collate(cases, mode=mode))
    P(f"samples={len(loader.dataset)}")

    for ep in range(a.epochs):
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for b in loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
                pred = model(b["x0"], b["coords"], b["static"], b["cond"], b["tau"])
                loss = wmse(pred, b["tgt"], cw)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
        if (ep + 1) % 20 == 0 or ep == a.epochs - 1:
            P(f"[E{ep + 1:03d}] trainloss={tot / nb:.4f} ({time.time() - t0:.0f}s)")

    torch.save({"model": model.state_dict(), "config": a.config, "cfg": cfg, "hp": HP,
                "fold": a.fold, "seed": a.seed, "test_keys": TEST, "normalizer": nz.state_dict()},
               os.path.join(out, "last.pt"))
    P(f"saved last-epoch ckpt -> {out}/last.pt")
    log.close()


if __name__ == "__main__":
    main()

