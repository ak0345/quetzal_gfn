"""
tempgain_guide.py -- a guide architecture that attacks the SATURATED-PRIOR CEILING.

Diagnosis recap (osim, beta=20, flow route):
  * Quetzal's proj_logits are the FINAL layer; gap between top-1 and top-2 logit
    is > 8 on ~72% of decisions (>16 on ~50%). The plain residual guide can only
    flip the ~18% of decisions where the gap < 4, so aggregate steering is capped.
  * Scaling the plain residual past ~2x drives validity to 0 (B_scale_sweep) --
    you cannot brute-force through the ceiling with a global gain.

MECHANISM (both learned per-state, so validity is protected):
    guided_logits = prior_logits / T(h)  +  g(h) * guide_residual(h)

  T(h)  : a learned per-STATE temperature on the prior. T > 1 SOFTENS, shrinking
          the huge top-1 gaps into a range the residual can contest; T < 1
          sharpens. Learned per state, so the net can choose T~1 where softening
          would break validity and T>1 only where there is room to steer.
  g(h)  : a learned per-STATE gain on the residual, so the guide amplifies its
          logits only where it is confident -- avoiding the global over-powering
          that collapsed validity when the residual was scaled uniformly.

Both heads are ZERO-INIT so the model STARTS at exactly the frozen prior's
behaviour (T=1, g=1) and only departs as training moves it.

=============================================================================
FIXED: the temperature head was inert in every run before this version.
=============================================================================
The previous parameterisation was

    T = 1 + (softplus(raw) - ln 2)          # claimed T >= 1

which actually spans (1 - ln 2, inf) = (0.307, inf) -- it does NOT enforce
T >= 1. The floor was supplied instead by `clamp(T, min=1.0)` in the forward
pass, and clamp has ZERO GRADIENT below the threshold. So the moment training
pushed the head under 1 it stopped receiving gradient and froze: the observed
T of 0.73-0.80, flat in the margin across every trained checkpoint, is a head
stuck in the clamp's dead zone, not a head that learned to sharpen. The
mechanism never engaged in either direction.

This version:
  * parameterises T in LOG space, T = exp(s * tanh(raw)), which is exactly 1 at
    raw=0, symmetric (softening and sharpening equally reachable and equally
    penalised, so a learned T<1 is a FINDING rather than an artifact), and
    bounded by construction in [1/e^s, e^s] so no clamp is needed anywhere;
  * has a non-vanishing gradient over that whole range, so the head can leave a
    region it has entered;
  * records per-state diagnostics (mean T, fraction softening, resulting margin
    compression) so "did the mechanism engage" is measured, not assumed;
  * makes forward() refuse to silently drop the temperature.

Old checkpoints are NOT loadable into this class (different parameter meaning);
see `migrate_note()`.
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# softplus^{-1}(1) so that a zero-init pre-activation gives softplus(...) = 1
_SOFTPLUS_INV_1 = math.log(math.e - 1.0)   # ~0.5413

# softplus^{-1}(1e-3): the offset that starts a positive-only log-temperature
# just above 0 (T ~ 1.001) while keeping the gradient non-zero there
_EPS_LOG_T = 1e-3
_SOFTPLUS_OFFSET = math.log(math.exp(_EPS_LOG_T) - 1.0)


class PriorTemperature(nn.Module):
    """Learned per-state temperature T(h) on the frozen prior logits.

    T = exp(max_log_T * tanh(raw)), so:
      raw = 0            -> T = 1 exactly (identity at init)
      raw > 0            -> T > 1, softening (the intended direction)
      raw < 0            -> T < 1, sharpening
      T in [e^-s, e^s]   -> bounded without a clamp, so the gradient never dies

    `allow_sharpening=False` restricts to T >= 1 the correct way -- through the
    parameterisation (softplus in log space), which stays differentiable
    everywhere -- rather than by clamping the output, which does not.
    """

    def __init__(self, d_model, hidden=128, max_log_T=math.log(4.0),
                 allow_sharpening=True):
        super().__init__()
        self.max_log_T = float(max_log_T)
        self.allow_sharpening = bool(allow_sharpening)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)      # raw = 0 -> T = 1 exactly

    def forward(self, h):
        raw = self.net(h).squeeze(-1)
        if self.allow_sharpening:
            log_T = self.max_log_T * torch.tanh(raw)
        else:
            # T >= 1 enforced through the PARAMETERISATION, not a clamp on the
            # output: softplus is positive and differentiable everywhere, so a
            # head pushed toward the floor keeps receiving gradient and can come
            # back -- which is precisely what the old clamp prevented.
            # The offset puts log_T at _EPS_LOG_T (not exactly 0) at init, so
            # T starts at 1.001 rather than 1: softplus never attains its
            # infimum, and buying exact identity here would mean a zero
            # gradient at the init point. Prefer allow_sharpening=True, whose
            # tanh parameterisation gives exact identity AND a live gradient.
            log_T = F.softplus(raw + _SOFTPLUS_OFFSET)
            log_T = self.max_log_T * torch.tanh(log_T / self.max_log_T)
        return torch.exp(log_T)


class ResidualGain(nn.Module):
    """Learned per-state gain g(h) >= 0 on the guide residual. Zero-init ->
    g = 1 everywhere at start (identity on the existing residual)."""

    def __init__(self, d_model, hidden=128, max_gain=None):
        super().__init__()
        self.max_gain = max_gain
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        # bias = softplus^{-1}(1) so g starts at exactly 1
        nn.init.constant_(self.net[-1].bias, _SOFTPLUS_INV_1)

    def forward(self, h):
        g = F.softplus(self.net(h).squeeze(-1))          # >= 0, starts at 1
        if self.max_gain is not None:
            # smooth ceiling: the scale sweep showed validity collapses past
            # ~4x, so an unbounded gain can walk the policy off the valid
            # region before the reward signal catches up
            g = self.max_gain * torch.tanh(g / self.max_gain)
        return g


class TempGainGuide(nn.Module):
    """Wraps an existing LogitGuide with a prior-temperature and a residual-gain.

        guided = prior_logits / T(h)[:, None]  +  g(h)[:, None] * residual(h)

    The base `guide` is your existing LogitGuide, so a trained one can be
    warm-started.
    """

    def __init__(self, base_guide, d_model, hidden=128,
                 use_temperature=True, use_gain=True,
                 max_log_T=math.log(4.0), allow_sharpening=True,
                 max_gain=None, collect_stats=True):
        super().__init__()
        self.guide = base_guide
        self.temp = PriorTemperature(d_model, hidden, max_log_T=max_log_T,
                                     allow_sharpening=allow_sharpening) \
            if use_temperature else None
        self.gain = ResidualGain(d_model, hidden, max_gain=max_gain) \
            if use_gain else None
        self.collect_stats = collect_stats
        self.last_stats = {}
        self._warned_forward = False

    # ---------------------------------------------------------------- core

    def guided_logits(self, prior_logits, h):
        """The full mechanism. No clamp: T is bounded by its parameterisation."""
        resid = self.guide(h)
        if self.gain is not None:
            g = self.gain(h)
            resid = g.unsqueeze(-1) * resid
        else:
            g = None
        if self.temp is not None:
            T = self.temp(h)
            prior_logits = prior_logits / T.unsqueeze(-1)
        else:
            T = None
        out = prior_logits + resid
        if self.collect_stats:
            self._record(T, g, resid, prior_logits, out)
        return out

    def forward(self, h, prior_logits=None):
        """With prior_logits: the full guided logits. Without: the gained
        residual ONLY -- the temperature cannot be applied, so a caller doing
        `prior_logits + guide(h)` silently gets a residual-plus-gain guide.
        That silent degradation is what made the last set of runs
        uninterpretable, so it now warns once."""
        if prior_logits is not None:
            return self.guided_logits(prior_logits, h)
        if self.temp is not None and not self._warned_forward:
            self._warned_forward = True
            warnings.warn(
                "TempGainGuide.forward(h) called without prior_logits: the "
                "temperature is NOT applied and this is a residual+gain guide. "
                "Call guided_logits(prior_logits, h) to use the mechanism.",
                RuntimeWarning, stacklevel=2)
        resid = self.guide(h)
        if self.gain is not None:
            resid = self.gain(h).unsqueeze(-1) * resid
        return resid

    def residual_only(self, h):
        """Explicit, non-warning version of the degraded path, for call sites
        that genuinely want the additive part (e.g. residual-norm diagnostics)."""
        resid = self.guide(h)
        if self.gain is not None:
            resid = self.gain(h).unsqueeze(-1) * resid
        return resid

    # ------------------------------------------------------------ telemetry

    @torch.no_grad()
    def _record(self, T, g, resid, softened_logits, out):
        st = {}
        if T is not None:
            st["T_mean"] = T.mean().item()
            st["T_min"] = T.min().item()
            st["T_max"] = T.max().item()
            # the number that says whether the mechanism engaged at all
            st["T_frac_softening"] = (T > 1.0).float().mean().item()
            st["T_frac_sharpening"] = (T < 1.0).float().mean().item()
        if g is not None:
            st["gain_mean"] = g.mean().item()
            st["gain_max"] = g.max().item()
        st["resid_norm"] = resid.norm(dim=-1).mean().item()
        # margin BEFORE and AFTER softening: the mechanism's whole purpose is to
        # shrink this, so log it rather than inferring it from the flip rate
        if softened_logits.shape[-1] >= 2:
            top2 = torch.topk(softened_logits, 2, dim=-1).values
            st["margin_after_T"] = (top2[:, 0] - top2[:, 1]).mean().item()
            t2o = torch.topk(out, 2, dim=-1).values
            st["margin_after_guide"] = (t2o[:, 0] - t2o[:, 1]).mean().item()
        self.last_stats = st

    @torch.no_grad()
    def margin_profile(self, prior_logits, h, bins=(0, 2, 4, 8, 16, 1e9)):
        """Mean T per prior-margin bin -- the figure that shows whether the head
        learned to soften WHERE THE PRIOR IS CONFIDENT (the design intent) or
        applied a flat temperature everywhere (no state-dependence learned)."""
        if self.temp is None:
            return {}
        top2 = torch.topk(prior_logits, 2, dim=-1).values
        margin = top2[:, 0] - top2[:, 1]
        T = self.temp(h)
        out = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (margin >= lo) & (margin < hi)
            if m.any():
                out[f"T_margin_{lo}_{hi}"] = T[m].mean().item()
                out[f"n_margin_{lo}_{hi}"] = int(m.sum().item())
        return out


def migrate_note():
    return (
        "Checkpoints trained with the previous PriorTemperature are not "
        "loadable here: `net` has the same shape but its output is now a "
        "log-temperature, so the same weights mean a different T. Those runs "
        "were also inert (clamped, zero-gradient), so there is nothing to "
        "carry over -- retrain, or load only the base guide and re-init the "
        "temperature head."
    )