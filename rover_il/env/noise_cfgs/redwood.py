import torch
from isaaclab.utils.noise import NoiseCfg
from isaaclab.utils import configclass



def RedwoodDepthNoise(depth: torch.Tensor, cfg) -> torch.Tensor:
    """
    Realistic Redwood-style noise model with support for operation modes.

    Args:
        depth: (B, 1, H, W) depth map in meters.
        cfg: RedwoodDepthNoiseCfg instance.

    Returns:
        (B, 1, H, W) noisy depth map or noise tensor (for 'abs' mode).
    """
    B, C, H, W = depth.shape
    device = depth.device

    # Spatial jitter
    rand_shift_y = (torch.randn(B, 1, H, W, device=device) * 2.0 * cfg.noise_multiplier).round().long()
    rand_shift_x = (torch.randn(B, 1, H, W, device=device) * 2.0 * cfg.noise_multiplier).round().long()

    yy = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)
    xx = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
    yy_shift = torch.clamp(yy + rand_shift_y, 0, H - 1)
    xx_shift = torch.clamp(xx + rand_shift_x, 0, W - 1)

    # Apply jitter
    shifted_depth = depth.gather(2, yy_shift).gather(3, xx_shift)

    # Depth-dependent std for exponential scaling
    std = cfg.scale * (torch.exp(cfg.exponent * shifted_depth.clamp(min=1.0)) - 1.0)
    noise = torch.randn_like(shifted_depth) * std * cfg.noise_multiplier

    # Apply operation
    if cfg.operation == "add":
        out = shifted_depth + noise
    elif cfg.operation in ["scale", "mult"]:
        out = shifted_depth * (1.0 + noise)
    elif cfg.operation == "abs":
        return noise
    else:
        raise ValueError(f"Unsupported noise operation: {cfg.operation}")

    return torch.clamp(out, 0.0, cfg.max_depth)


@configclass
class RedwoodDepthNoiseCfg(NoiseCfg):
    """Redwood-style exponential noise for depth sensing with scaling support."""
    func: callable = RedwoodDepthNoise
    scale: float = 0.1
    exponent: float = 0.3
    max_depth: float = 10.0
    noise_multiplier: float = 1.0
    operation: str = "add"  # "scale", "add", or "abs"
