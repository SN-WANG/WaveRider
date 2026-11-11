import torch

CHANNELS = ["v_z", "v_r", "p", "T", "rho"]
LOG_CHANNELS = {"p", "T", "rho"}
LOG_IDX = [i for i, c in enumerate(CHANNELS) if c in LOG_CHANNELS]
LOG_CLAMP = 1e-12


def phys_to_transformed(fields: torch.Tensor) -> torch.Tensor:
    out = fields.clone()
    for i in LOG_IDX:
        out[..., i] = torch.log(fields[..., i].clamp_min(LOG_CLAMP))
    return out


def transformed_to_phys(t: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    for i in LOG_IDX:
        out[..., i] = torch.exp(t[..., i])
    return out


class Normalizer:

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, channels=CHANNELS, eps: float = 1e-8):
        self.mean = mean.float()
        self.std = std.float().clamp_min(eps)
        self.channels = list(channels)

    def normalize(self, fields_phys: torch.Tensor) -> torch.Tensor:
        t = phys_to_transformed(fields_phys)
        return (t - self.mean.to(t)) / self.std.to(t)

    def denormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        t = x_norm * self.std.to(x_norm) + self.mean.to(x_norm)
        return transformed_to_phys(t)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "channels": self.channels}

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return cls(d["mean"], d["std"], d.get("channels", CHANNELS))

