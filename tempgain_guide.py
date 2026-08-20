"""
Guide architecture with a learned per-state temperature on the prior and a
learned per-state gain on the residual.

    guided_logits = prior_logits / T(h)  +  g(h) * residual(h)

The residual guide competes against logits whose top-1 margin averages 16.4,
with the margin above 8 on ~72% of decisions. This architecture is meant to
change the terms of that competition rather than the size of the residual:

  T(h)  A per-state temperature. T > 1 compresses the margin so a residual of
        unchanged magnitude can contest decisions it could not otherwise reach;
        T < 1 sharpens. Learned per state, because a uniform softening applies
        where it helps and where it destroys validity alike.
  g(h)  A per-state gain, so the residual is amplified only where the guide is
        confident. A constant multiplier instead raises effect size up to
        roughly 4x and then reverses as validity collapses.

Both heads start at the identity (T = 1, g = 1), so training begins at the
frozen prior's behaviour.

The temperature was inactive in the runs reported in the paper
--------------------------------------------------------------
Those runs used the parameterisation

    T = 1 + (softplus(raw) - ln 2)

which spans (1 - ln 2, inf) = (0.307, inf) and does not enforce T >= 1. The
floor came instead from clamp(T, min=1.0) in the forward pass, and clamp has
zero gradient below its threshold: once training pushed the head under 1 it
stopped receiving gradient and froze. The T of 0.73-0.80 read off every trained
checkpoint, flat in the margin, is a head stuck in that dead zone rather than
one that learned to sharpen, and the effective temperature was 1 at every state.
Runs labelled TEMPGAIN therefore report a gain-scaled residual guide, not a test
of prior softening.

This version parameterises T in log space as T = exp(s * tanh(raw)): exactly 1
at raw = 0, symmetric so softening and sharpening are equally reachable and
equally penalised (a learned T < 1 becomes a finding rather than an artifact),
and bounded in [e^-s, e^s] by construction so no clamp is needed and the
gradient never dies. It also records per-state diagnostics -- mean T, fraction
softening, resulting margin compression -- so whether the mechanism engaged is
measured rather than assumed, and refuses to drop the temperature silently.

Checkpoints from the previous parameterisation are not loadable here; see
`migrate_note()`.
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

    `allow_sharpening=False` restricts to T >= 1 through the parameterisation
    (softplus in log space), which stays differentiable everywhere, rather than
    by clamping the output, which does not.
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
            # T >= 1 through the parameterisation rather than a clamp on the
            # output: softplus is positive and differentiable everywhere, so a
            # head pushed toward the floor keeps receiving gradient and can
            # leave again.
            #
            # The offset puts log_T at _EPS_LOG_T rather than exactly 0 at
            # initialisation, so T starts at 1.001: softplus never attains its
            # infimum, and exact identity here would cost a zero gradient at
            # the initialisation point. allow_sharpening=True is preferable --
            # its tanh parameterisation gives exact identity and a live
            # gradient.
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
            # Smooth ceiling. The residual-scale sweep shows validity
            # collapsing past ~4x, so an unbounded gain can walk the policy off
            # the valid region faster than the reward signal compensates.
            g = self.max_gain * torch.tanh(g / self.max_gain)
        return g


class TempGainGuide(nn.Module):
    """Wraps an existing LogitGuide with a prior-temperature and a residual-gain.

        guided = prior_logits / T(h)[:, None]  +  g(h)[:, None] * residual(h)

    The base `guide` is a LogitGuide, so an already-trained residual can be
    warm-started and only the two heads need to move.
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
        """The full mechanism. No clamp -- T is bounded by its parameterisation."""
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
        """With prior_logits, the full guided logits. Without, the gained
        residual only: the temperature cannot be applied, so a caller writing
        `prior_logits + guide(h)` gets a residual-plus-gain guide instead of
        this architecture. That degradation is silent and is what made the
        earlier runs report the wrong mechanism, so it warns once."""
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
        """The additive part alone, without the warning, for call sites that
        genuinely want it -- residual-norm diagnostics, for instance."""
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
            # whether the mechanism engaged at all
            st["T_frac_softening"] = (T > 1.0).float().mean().item()
            st["T_frac_sharpening"] = (T < 1.0).float().mean().item()
        if g is not None:
            st["gain_mean"] = g.mean().item()
            st["gain_max"] = g.max().item()
        st["resid_norm"] = resid.norm(dim=-1).mean().item()
        # Margin before and after softening. Shrinking it is the mechanism's
        # purpose, so record it directly rather than inferring it from the
        # flip rate.
        if softened_logits.shape[-1] >= 2:
            top2 = torch.topk(softened_logits, 2, dim=-1).values
            st["margin_after_T"] = (top2[:, 0] - top2[:, 1]).mean().item()
            t2o = torch.topk(out, 2, dim=-1).values
            st["margin_after_guide"] = (t2o[:, 0] - t2o[:, 1]).mean().item()
        self.last_stats = st

    @torch.no_grad()
    def margin_profile(self, prior_logits, h, bins=(0, 2, 4, 8, 16, 1e9)):
        """Mean T per prior-margin bin: whether the head learned to soften
        where the prior is confident, which is the design intent, or applied a
        flat temperature everywhere, meaning no state dependence was learned."""
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