"""
tempgain_guide.py -- a guide architecture that attacks the SATURATED-PRIOR CEILING.

Diagnosis recap (osim, beta=20, flow route):
  * Quetzal's proj_logits are the FINAL layer; gap between top-1 and top-2 logit
    is > 8 on ~72% of decisions (>16 on ~50%). The plain residual guide can only
    flip the ~18% of decisions where the gap < 4, so aggregate steering is capped.
  * Scaling the plain residual past ~2x drives validity to 0 (B_scale_sweep) --
    you cannot brute-force through the ceiling with a global gain.

FIX (both mechanisms, learned per-state so validity is protected):
    guided_logits = prior_logits / T(h)  +  g(h) * guide_residual(h)

  T(h) >= 1  : a learned per-STATE temperature that SOFTENS the prior, shrinking
               the huge top-1 gaps into a range the residual can contest. Because
               T is learned per state, the net can choose T~1 (no softening) on
               decisions where softening would break validity, and T>1 only where
               there is room to steer.
  g(h) >= 0  : a learned per-STATE gain on the residual, so the guide amplifies
               its logits only where it is confident -- avoiding the global
               over-powering that collapsed validity when the residual was scaled
               uniformly.

Both heads are ZERO-INIT so the model STARTS at exactly the current behavior
(T=1, g=1) and only departs as training moves it -- no cold-start regression.

    T(h) = 1 + softplus(w_T . h + b_T)            (>= 1, starts at 1)
    g(h) = softplus(w_g . h + b_g + SOFTPLUS_INV_1)  (>= 0, starts at 1)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus^{-1}(1) so that a zero-init pre-activation gives softplus(...) = 1
_SOFTPLUS_INV_1 = math.log(math.e - 1.0)   # ~0.5413


class PriorTemperature(nn.Module):
    """Learned per-state temperature T(h) >= 1 that softens the frozen prior
    logits. Zero-init -> T = 1 everywhere at start (identity)."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # zero-init final layer so raw pre-activation = 0 -> softplus(0)=~0.69,
        # but we want T=1 at start: subtract softplus(0) then +0 via offset below.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        # T = 1 + softplus(raw); raw starts at 0 -> softplus(0)=0.693 -> T=1.69.
        # To start EXACTLY at T=1 we subtract the constant softplus(0):
        raw = self.net(h).squeeze(-1)
        return 1.0 + (F.softplus(raw) - math.log(2.0))   # softplus(0)=ln2


class ResidualGain(nn.Module):
    """Learned per-state gain g(h) >= 0 on the guide residual. Zero-init ->
    g = 1 everywhere at start (identity on the existing residual)."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        # bias = softplus^{-1}(1) so g starts at exactly 1
        nn.init.constant_(self.net[-1].bias, _SOFTPLUS_INV_1)

    def forward(self, h):
        return F.softplus(self.net(h).squeeze(-1))       # >= 0, starts at 1


class TempGainGuide(nn.Module):
    """Wraps an existing LogitGuide with a prior-temperature and a residual-gain.

    Produces the GUIDED logits directly from (prior_logits, h):
        guided = prior_logits / T(h)[:,None]  +  g(h)[:,None] * residual(h)

    Use this in place of the raw `prior_logits + guide(h)` expression. The base
    `guide` is your existing LogitGuide (so you can warm-start from a trained one).
    """
    def __init__(self, base_guide, d_model, hidden=128,
                 use_temperature=True, use_gain=True):
        super().__init__()
        self.guide = base_guide
        self.temp = PriorTemperature(d_model, hidden) if use_temperature else None
        self.gain = ResidualGain(d_model, hidden) if use_gain else None

    def guided_logits(self, prior_logits, h):
        resid = self.guide(h)
        if self.gain is not None:
            resid = self.gain(h).unsqueeze(-1) * resid
        if self.temp is not None:
            prior_logits = prior_logits / self.temp(h).unsqueeze(-1).clamp(min=1.0)
        return prior_logits + resid

    def forward(self, h):
        # convenience: returns just the (gained) residual, so code paths that do
        # `prior_logits + guide(h)` still work for the gain part. But prefer
        # guided_logits(...) so the temperature is applied too.
        resid = self.guide(h)
        if self.gain is not None:
            resid = self.gain(h).unsqueeze(-1) * resid
        return resid