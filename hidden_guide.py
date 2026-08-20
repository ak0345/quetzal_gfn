"""
Guide architecture that injects on the hidden state, before the frozen
projection.

The residual guide adds to proj_logits(h). Those logits are large (norm ~6841)
and decisive: the top-1 margin exceeds 8 on ~72% of decisions, so a residual of
realistic norm cannot change the argmax there.

This guide perturbs h instead and lets the frozen proj_logits map the result to
logits. Because proj_logits amplifies displacements along the directions it
uses, a small well-aligned change to h produces a large change in logit space --
empirically a delta of norm 0.1 moves the logits by ~46, against ~1.3 for an
output residual of the same norm.

    residual :  guided_logits = proj_logits(h) + guide(h)
    hidden   :  guided_logits = proj_logits(h + delta(h))

delta is zero-initialised, so the guided logits equal the prior's exactly at
initialisation and training begins at the frozen model.

Unlike LogitGuide, which only sees h, this guide needs a reference to the
prior's proj_logits to run its forward pass.
"""
import torch
import torch.nn as nn


class HiddenGuide(nn.Module):
    """Perturbs the hidden state before the frozen projection.

        guided_logits(h) = proj_logits(h + delta(h))  [+ optional out_residual(h)]

    delta is zero-initialised, so this starts at the frozen prior's logits.
    """
    def __init__(self, d_model, proj_logits, hidden=512, layers=2,
                 vocab_size=128, also_output_residual=False):
        super().__init__()
        self.d_model = d_model
        # A reference to the projection, which the prior has already frozen; it
        # is deliberately not registered as a trainable parameter here.
        self._proj = proj_logits
        self.vocab_size = vocab_size

        blocks = []
        din = d_model
        for _ in range(max(layers - 1, 0)):
            blocks += [nn.Linear(din, hidden), nn.SiLU()]
            din = hidden
        blocks += [nn.Linear(din, d_model)]   # the output is a hidden-state delta
        self.delta = nn.Sequential(*blocks)
        # zero-init the last layer so delta(h) = 0 at initialisation
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        # optional additive residual on the output logits, also zero-init
        self.out_residual = None
        if also_output_residual:
            self.out_residual = nn.Sequential(
                nn.Linear(d_model, hidden), nn.SiLU(),
                nn.Linear(hidden, vocab_size))
            nn.init.zeros_(self.out_residual[-1].weight)
            nn.init.zeros_(self.out_residual[-1].bias)

    def guided_logits(self, h):
        """Hidden state in, guided logits out (the projection is already applied)."""
        d = self.delta(h)
        logits = self._proj(h + d)
        if self.out_residual is not None:
            logits = logits + self.out_residual(h)
        return logits

    def forward(self, h):
        """Return guided_logits - prior_logits, so call sites expecting an
        additive residual still compose to the guided logits via
        `prior + guide(h)`. Prefer guided_logits(h) where possible."""
        with torch.no_grad():
            prior_logits = self._proj(h)
        return self.guided_logits(h) - prior_logits