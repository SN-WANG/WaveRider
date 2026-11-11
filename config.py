import os

ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.environ.get("WR_DATA_DIR", os.path.join(ROOT, "data"))
DOE_CSV = os.environ.get("WR_DOE_CSV", os.path.join(ROOT, "doe_combined.csv"))
CKPT_DIR = os.environ.get("WR_CKPT_DIR", os.path.join(ROOT, "checkpoints"))

N_FOLDS = 5

CHANNELS = ["v_z", "v_r", "p", "T", "rho"]

RGAS_N2 = 297.0
GAMMA = 1.4
P_LOW = 1000.0
WV_ALIGN = 1.0

HP = dict(
    hidden=192, depth=6, num_slices=32, num_heads=8, dropout=0.05,
    epochs=250, lr=2e-4, wd=1e-5, batch_size=32,
    cw=[2.0, 2.0, 1.0, 1.0, 1.0],
)

MODEL_CONFIGS = {
    "transolver":    dict(model="transolver", mode="direct", waverider=False),
    "waverider":     dict(model="transolver", mode="direct", waverider=True),
    "fixed_coord":   dict(model="transolver", mode="direct", waverider=True, wv_fixed=True),
    "head_reinject": dict(model="transolver", mode="direct", waverider=True, head_reinject=True),
    "transolver_ar": dict(model="transolver", mode="ar",     waverider=False),
    "deeponet":      dict(model="deeponet",   mode="direct", waverider=False),
    "deeponet_ar":   dict(model="deeponet",   mode="ar",     waverider=False),
}
