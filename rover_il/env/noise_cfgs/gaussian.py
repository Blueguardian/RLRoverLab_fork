from isaaclab.utils.noise import NoiseCfg
from isaaclab.utils.noise import noise_model
from isaaclab.utils.configclass import configclass
import torch

def safe_gaussian_noise(data: torch.Tensor, cfg) -> torch.Tensor:
    noise = cfg.mean + cfg.std * torch.randn_like(data)

    if cfg.operation == "add":
        out = data + noise
    elif cfg.operation == "scale":
        out = data * (1.0 + noise)
    elif cfg.operation == "abs":
        return noise
    else:
        raise ValueError(f"Unknown operation: {cfg.operation}")

    # Clamp output to prevent overflows (assume 8-bit image range)
    return torch.clamp(out, 0.0, 255.0)

@configclass
class GaussianImageNoiseCfg(NoiseCfg):
    func = safe_gaussian_noise
    std: float = 10.0
    mean: float = 5.0
    operation: str = "add"