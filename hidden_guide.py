"""
hidden_guide.py -- Fix B: inject the guide BEFORE the final projection.

THE PROBLEM (the ceiling): the plain guide adds a residual to proj_logits(h),
whose output logits are enormous (norm ~6841) and decisive (top-1 gap > 8 on 72%
of decisions). A residual of realistic norm (~5) is a rounding error at that scale
and cannot flip the argmax on high-gap decisions.

THE IDEA: perturb the HIDDEN STATE h instead, then let the frozen proj_logits map
it to logits. proj_logits is a learned linear map; a small, well-aligned change to
h is AMPLIFIED by proj_logits into a large, correctly-shaped change in logits --
because it moves along the directions proj_logits actually uses. We are no longer
fighting the output magnitude; we are steering the input to the saturating layer.

    plain :  guided_logits = proj_logits(h) + guide(h)
    hidden:  guided_logits = proj_logits(h + delta(h))          [fix B]

delta(h) is a learned residual on the hidden state (same dim as h). Zero-init so
delta=0 at start -> guided_logits == proj_logits(h) (the frozen prior) -> identity,
no cold-start regression.

INTEGRATION: this guide needs the prior's proj_logits to run its forward, so unlike
LogitGuide (which only sees h), it must be given a reference to proj_logits.
"""
import torch
import torch.nn as nn


class HiddenGuide(nn.Module):
    """Guide that perturbs the hidden state before the frozen projection.

        guided_logits(h) = proj_logits( h + delta(h) )  [ + optional out_residual(h) ]

    delta is zero-init -> starts at the exact frozen-prior logits.
    """
    def __init__(self, d_model, proj_logits, hidden=512, layers=2,
                 vocab_size=128, also_output_residual=False):
        super().__init__()
        self.d_model = d_model
        # keep a reference to the FROZEN projection (not registered as a parameter
        # to train; we rely on it already being frozen in the prior).
        self._proj = proj_logits
        self.vocab_size = vocab_size

        blocks = []
        din = d_model
        for _ in range(max(layers - 1, 0)):
            blocks += [nn.Linear(din, hidden), nn.SiLU()]
            din = hidden
        blocks += [nn.Linear(din, d_model)]   # output is a HIDDEN-STATE delta
        self.delta = nn.Sequential(*blocks)
        # zero-init the last layer so delta(h)=0 at start -> identity
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        # optional small output-logit residual (belt & braces); zero-init too
        self.out_residual = None
        if also_output_residual:
            self.out_residual = nn.Sequential(
                nn.Linear(d_model, hidden), nn.SiLU(),
                nn.Linear(hidden, vocab_size))
            nn.init.zeros_(self.out_residual[-1].weight)
            nn.init.zeros_(self.out_residual[-1].bias)

    def guided_logits(self, h):
        """Takes the hidden state h, returns GUIDED LOGITS (proj already applied)."""
        d = self.delta(h)
        logits = self._proj(h + d)
        if self.out_residual is not None:
            logits = logits + self.out_residual(h)
        return logits

    def forward(self, h):
        """For compatibility with call sites that expect a residual: return
        (guided_logits - prior_logits) so `prior + guide(h)` still yields the
        guided logits. Prefer guided_logits(h) directly."""
        with torch.no_grad():
            prior_logits = self._proj(h)
        return self.guided_logits(h) - prior_logits