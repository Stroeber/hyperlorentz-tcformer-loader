import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


class CutFillConfig:
    __slots__ = ("min_frac", "max_frac", "fill_value", "loss_on_full")
    def __init__(self, min_frac=0.1, max_frac=0.25, fill_value=0.0, loss_on_full=False):
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)
        self.fill_value = float(fill_value)
        self.loss_on_full = bool(loss_on_full)
    def __repr__(self):
        return (f"CutFillConfig(min_frac={self.min_frac}, max_frac={self.max_frac}, "
                f"fill_value={self.fill_value}, loss_on_full={self.loss_on_full})")


def _rand_span(T, min_frac, max_frac):
    min_len = max(1, int(T * min_frac))
    max_len = max(min_len, int(T * max_frac))
    span_len = torch.randint(min_len, max_len + 1, ()).item()
    start = torch.randint(0, max(1, T - span_len + 1), ()).item()
    return start, start + span_len


def apply_cut_and_fill(x: torch.Tensor, cfg: CutFillConfig):
    """
    EEG cut-and-fill for x shaped (B, 1, C, T).
      - Masks one contiguous time span per batch item, across all channels.
      - Fills with cfg.fill_value (scalar or Tensor broadcastable to (1,1,C,1)).

    Returns:
      x_masked: (B, 1, C, T)
      mask:     (B, 1, C, T) boolean (True where masked)
      spans:    list of (start, end) for each item
    """

    assert x.ndim == 4 and x.size(1) == 1, "Expected x with shape (B, 1, C, T)."
    B, _, C, T = x.shape
    device, dtype = x.device, x.dtype

    x_masked = x.clone()
    mask = torch.zeros((B, 1, C, T), dtype=torch.bool, device=device)
    spans = []

    # prepare a broadcastable filler of shape (1, 1, C, 1)
    fv = cfg.fill_value
    if isinstance(fv, torch.Tensor):
        filler = fv.to(device=device, dtype=dtype).view(1, 1, -1, 1)  # expect C or 1*C*1
        if filler.size(2) not in (1, C):
            raise ValueError(f"fill_value Tensor must have C={C} or 1 channel, got {filler.size(2)}")
    else:
        filler = torch.as_tensor(fv, device=device, dtype=dtype).view(1, 1, 1, 1)

    for b in range(B):
        s, e = _rand_span(T, cfg.min_frac, cfg.max_frac)
        spans.append((s, e))
        x_masked[b, :, :, s:e] = filler
        mask[b, :, :, s:e] = True

    return x_masked, mask, spans


def apply_multi_span_mask(x: torch.Tensor, cfg: CutFillConfig, n_spans: int = 3):
    """
    Multi-span masking for x shaped (B, 1, C, T).
    Masks multiple non-overlapping contiguous time spans per batch item.
    
    Returns:
      x_masked: (B, 1, C, T)
      mask:     (B, 1, C, T) boolean (True where masked)
      spans:    list of list of (start, end) for each item
    """
    assert x.ndim == 4 and x.size(1) == 1, "Expected x with shape (B, 1, C, T)."
    B, _, C, T = x.shape
    device, dtype = x.device, x.dtype

    x_masked = x.clone()
    mask = torch.zeros((B, 1, C, T), dtype=torch.bool, device=device)
    all_spans = []

    # Adjust fractions per span
    span_min_frac = cfg.min_frac / n_spans
    span_max_frac = cfg.max_frac / n_spans

    fv = cfg.fill_value
    if isinstance(fv, torch.Tensor):
        filler = fv.to(device=device, dtype=dtype).view(1, 1, -1, 1)
    else:
        filler = torch.as_tensor(fv, device=device, dtype=dtype).view(1, 1, 1, 1)

    for b in range(B):
        spans = []
        masked_positions = set()
        for _ in range(n_spans):
            # Try to find a non-overlapping span
            for attempt in range(10):
                s, e = _rand_span(T, span_min_frac, span_max_frac)
                span_set = set(range(s, e))
                if not span_set & masked_positions:
                    spans.append((s, e))
                    masked_positions.update(span_set)
                    x_masked[b, :, :, s:e] = filler
                    mask[b, :, :, s:e] = True
                    break
        all_spans.append(spans)

    return x_masked, mask, all_spans


def apply_channel_dropout(x: torch.Tensor, drop_prob: float = 0.1):
    """
    Randomly drop entire channels for x shaped (B, 1, C, T).
    
    Returns:
      x_dropped: (B, 1, C, T)
      mask:      (B, 1, C, T) boolean (True where dropped)
    """
    assert x.ndim == 4 and x.size(1) == 1, "Expected x with shape (B, 1, C, T)."
    B, _, C, T = x.shape
    device = x.device

    # Generate channel dropout mask (B, 1, C, 1) -> broadcast to T
    drop_mask = torch.rand(B, 1, C, 1, device=device) < drop_prob
    drop_mask = drop_mask.expand(-1, -1, -1, T)

    x_dropped = x.clone()
    x_dropped[drop_mask] = 0.0

    return x_dropped, drop_mask


def apply_combined_augmentation(x: torch.Tensor, cfg: CutFillConfig,
                                 n_spans: int = 2, channel_drop_prob: float = 0.1):
    """
    Apply both multi-span masking and channel dropout.
    
    Returns:
      x_aug: augmented input
      mask:  combined mask (True where masked/dropped)
    """
    x_aug, span_mask, spans = apply_multi_span_mask(x, cfg, n_spans=n_spans)
    x_aug, chan_mask = apply_channel_dropout(x_aug, drop_prob=channel_drop_prob)
    combined_mask = span_mask | chan_mask
    return x_aug, combined_mask, spans


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
    """
    InfoNCE contrastive loss for representation learning.
    z1, z2: (B, D) embeddings from two augmented views of the same samples.
    
    Positive pairs: (z1[i], z2[i])
    Negative pairs: all other combinations
    """
    B = z1.size(0)
    device = z1.device

    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Compute similarity matrix
    sim = torch.mm(z1, z2.t()) / temperature  # (B, B)

    # Labels: diagonal entries are positive pairs
    labels = torch.arange(B, device=device)

    # InfoNCE loss (symmetric)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)

    return (loss_12 + loss_21) / 2


def contrastive_loss_hyper(manifold, z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1):
    """
    InfoNCE contrastive loss for representation learning.
    z1, z2: (B, D) embeddings from two augmented views of the same samples.

    Positive pairs: (z1[i], z2[i])
    Negative pairs: all other combinations
    """
    B = z1.size(0)
    device = z1.device

    # Compute similarity matrix
    sim = manifold.csqdist(z1.permute(1,0,2), z2.permute(1,0,2)).mean(0) / temperature  # (B, B)

    # Labels: diagonal entries are positive pairs
    labels = torch.arange(B, device=device)

    # InfoNCE loss (symmetric)
    loss_12 = F.cross_entropy(sim, labels)
    loss_21 = F.cross_entropy(sim.t(), labels)

    return (loss_12 + loss_21) / 2

def temporal_order_loss(encoder: torch.nn.Module, x: torch.Tensor, n_segments: int = 4):
    """
    Temporal order prediction: shuffle segments and predict correct order.
    x: (B, 1, C, T)
    
    Returns:
      loss: cross-entropy loss for order prediction
      accuracy: classification accuracy
    """
    B, _, C, T = x.shape
    device = x.device

    seg_len = T // n_segments
    if seg_len * n_segments != T:
        # Trim to make divisible
        x = x[..., :seg_len * n_segments]
        T = seg_len * n_segments

    # Split into segments: (B, n_segments, 1, C, seg_len)
    segments = x.view(B, 1, C, n_segments, seg_len).permute(0, 3, 1, 2, 4)

    # Generate random permutations for each batch
    perm_indices = torch.stack([torch.randperm(n_segments, device=device) for _ in range(B)])

    # Shuffle segments
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_segments)
    shuffled = segments[batch_idx, perm_indices]  # (B, n_segments, 1, C, seg_len)

    # Reconstruct shuffled sequence
    x_shuffled = shuffled.permute(0, 2, 3, 1, 4).reshape(B, 1, C, T)

    return x_shuffled, perm_indices


def variance_floor_loss_eeg(y_hat: torch.Tensor, floor: float = 0.5):
    """
    Hinge penalty if per-sample std across (C,T) falls below `floor`.
    y_hat: (B, 1, C, T)
    """
    std = y_hat.std(dim=(-2, -1), unbiased=False)     # (B, 1)
    return (torch.abs(floor - std))**2


def reconstruction_loss(x: torch.Tensor, x_hat: torch.Tensor,
                            mask: torch.Tensor | None = None, reduction: str = "mean"):
    """
    Mask-aware MSE for arbitrary shapes. If `mask` is provided, it must be
    broadcastable to `x` (e.g., mask same shape as x or with singleton dims).

    - reduction='mean': sum(diff*mask) / (#masked elements)
    - reduction='sum':  sum(diff*mask)
    - reduction='none': elementwise (same shape as x)

    Works for:
      (B, T, D), (B, 1, C, T), etc.
    """

    # var_reg = variance_floor_loss_eeg(x_hat, floor=0.5)

    if mask is None:
        return F.mse_loss(x_hat, x, reduction=reduction)  # + 5*var_reg.sum()

    diff2 = (x_hat - x) ** 2
    m = mask.to(dtype=diff2.dtype)
    diff2 = diff2 * m

    if reduction == "none":
        return diff2
    elif reduction == "sum":
        return diff2.sum()
    elif reduction == "mean":
        denom = m.sum()
        # avoid div-by-zero if mask is empty (shouldn't happen)
        return diff2.sum() / torch.clamp(denom, min=1.0)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
