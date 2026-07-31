from typing import Iterable, Dict

from reconstruction.losses import *
from reconstruction.model import TimeSeriesPretrainer


def slice_time_series(x, n_windows):
    """
    Slice a batched time series (batch, time, channel) into N equal-size windows
    with minimal overlap along the time dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, time, channel).
    n_windows : int
        Number of windows to generate.

    Returns
    -------
    windows : torch.Tensor
        Output tensor of shape (batch, window_size * n_windows, channel).
    """
    if x.ndim != 3:
        raise ValueError("Input must have shape (batch, time, channel)")

    B, T, C = x.shape

    if n_windows <= 0 or n_windows > T:
        raise ValueError("n_windows must be between 1 and the time dimension size")

    # Window size = ceil(T / N) to cover the full time series
    win_size = (T + n_windows - 1) // n_windows  # Equivalent to ceil(T / n_windows)

    # Stride so windows are evenly spread
    stride = (T - win_size) / max(n_windows - 1, 1)


    windows = []
    for i in range(n_windows):
        start = int(round(i * stride))
        end = start + win_size
        windows.append(x[:, start:end, :].unsqueeze(1))  # shape: (batch, window_size, channel)

    # Concatenate along the time dimension
    return torch.cat(windows, dim=1).reshape(B*n_windows, -1, C)


class TrainConfig:
    __slots__ = ("task_mix_recon", "cutfill_cfg", "grad_clip", 
                 "contrastive_weight", "n_mask_spans", "channel_drop_prob")
    def __init__(self, task_mix_recon=0.5, cutfill_cfg=None, grad_clip=1.0,
                 contrastive_weight=0.1, n_mask_spans=2, channel_drop_prob=0.1):
        self.task_mix_recon = float(task_mix_recon)
        self.cutfill_cfg = cutfill_cfg if cutfill_cfg is not None else CutFillConfig()
        self.grad_clip = None if grad_clip is None else float(grad_clip)
        self.contrastive_weight = float(contrastive_weight)
        self.n_mask_spans = int(n_mask_spans)
        self.channel_drop_prob = float(channel_drop_prob)
    def __repr__(self):
        return (f"TrainConfig(task_mix_recon={self.task_mix_recon}, "
                f"cutfill_cfg={self.cutfill_cfg}, grad_clip={self.grad_clip}, "
                f"contrastive_weight={self.contrastive_weight}, n_mask_spans={self.n_mask_spans}, "
                f"channel_drop_prob={self.channel_drop_prob})")


def train_epoch(model: TimeSeriesPretrainer,
                loader: Iterable,
                optimizer: torch.optim.Optimizer,
                device: str,
                cfg: TrainConfig,
                subject: bool,
                scheduler=None) -> Dict[str, float]:
    """
    Train for one epoch with improved masking strategies and optional contrastive loss.
    
    Scheduler is now stepped per-epoch (caller responsibility), not per-batch.
    """
    model.train()
    total, n = 0.0, 0
    total_recon, total_cutfill, total_contrastive = 0.0, 0.0, 0.0
    
    for x in loader:  # x: (B, T, D)

        if subject:
            x_orig, x_sub, yb = x
            x_sub = x_sub.to(device)
        else:
            x_orig, yb = x

        x_orig = x_orig.to(device)

        if len(x_orig.shape) == 3:
            x_orig = x_orig.unsqueeze(1)

        do_recon = (torch.rand(()) < cfg.task_mix_recon)
        optimizer.zero_grad(set_to_none=True)

        if do_recon:
            # Plain reconstruction with optional channel dropout
            if cfg.channel_drop_prob > 0:
                x_in, _ = apply_channel_dropout(x_orig, drop_prob=cfg.channel_drop_prob)
            else:
                x_in = x_orig
            if subject:
                x_hat = model(x_in, x_sub)
            else:
                x_hat = model(x_in)
            loss = reconstruction_loss(x_hat, x_orig)
            total_recon += loss.item()
        else:
            # Multi-span cut-and-fill with channel dropout
            x_in, mask, _ = apply_combined_augmentation(
                x_orig, cfg.cutfill_cfg, 
                n_spans=cfg.n_mask_spans, 
                channel_drop_prob=cfg.channel_drop_prob
            )
            if subject:
                x_hat = model(x_in, x_sub)
            else:
                x_hat = model(x_in)
            if cfg.cutfill_cfg.loss_on_full:
                loss = reconstruction_loss(x_orig, x_hat)
            else:
                loss = reconstruction_loss(x_orig, x_hat, mask=mask)
            total_cutfill += loss.item()

        # Optional contrastive loss between two augmented views
        if cfg.contrastive_weight > 0:
            # Create two different augmented views
            x_aug1, _, _ = apply_combined_augmentation(
                x_orig, cfg.cutfill_cfg,
                n_spans=cfg.n_mask_spans,
                channel_drop_prob=cfg.channel_drop_prob
            )
            x_aug2, _, _ = apply_combined_augmentation(
                x_orig, cfg.cutfill_cfg,
                n_spans=cfg.n_mask_spans,
                channel_drop_prob=cfg.channel_drop_prob
            )

            z1 = model.encoder(slice_time_series(model.to_hyperbolic(model.processor(x_aug1, x_sub).squeeze()), model.windows))
            z2 = model.encoder(slice_time_series(model.to_hyperbolic(model.processor(x_aug2, x_sub).squeeze()), model.windows))
            
            # Flatten embeddings if needed
            if z1.ndim > 2:
                z1 = z1[..., 0:].view(z1.size(0), -1)
                z2 = z2[..., 0:].view(z2.size(0), -1)
            
            contrastive = contrastive_loss(z1, z2)
            loss = loss + cfg.contrastive_weight * contrastive
            total_contrastive += contrastive.item()

        loss.backward()
        
        # Optional gradient clipping
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        optimizer.step()

        total += loss.item()
        n += 1

    # Note: scheduler.step() should be called per-epoch in the main training loop

    return {
        "loss": total / max(1, n),
        "recon_loss": total_recon / max(1, n),
        "cutfill_loss": total_cutfill / max(1, n),
        "contrastive_loss": total_contrastive / max(1, n),
        "steps": n,
    }


@torch.no_grad()
def evaluate(model: TimeSeriesPretrainer, loader: Iterable, device: str,
             cutfill_eval_cfg: CutFillConfig, subject: bool) -> Dict[str, float]:
    model.eval()
    total_recon, total_cutfill = 0.0, 0.0
    n1, n2 = 0, 0
    for x in loader:

        if subject:
            x_orig, x_sub, yb = x
            x_sub = x_sub.to(device)
        else:
            x_orig, yb = x

        x_orig = x_orig.to(device)


        if len(x_orig.shape) == 3:
            x_orig = x_orig.unsqueeze(1)

        # Full reconstruction error
        if subject:
            x_hat = model(x_orig, x_sub)
        else:
            x_hat = model(x_orig)

        total_recon += reconstruction_loss(x_orig, x_hat).item()
        n1 += 1

        # Cut-and-fill evaluation: compute loss only over masked region
        x_in, mask, _ = apply_cut_and_fill(x_orig, cutfill_eval_cfg)
        if subject:
            x_hat2 = model(x_in, x_sub)
        else:
            x_hat2 = model(x_in)
        total_cutfill += reconstruction_loss(x_orig, x_hat2, mask=mask).item()
        n2 += 1

    return {
        "val_recon_mse": total_recon / max(1, n1),
        "val_cutfill_mse": total_cutfill / max(1, n2),
    }
