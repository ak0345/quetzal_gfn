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

Optionally BOTH (hidden delta + a small output residual) if you want belt & braces,
but the point of B is the hidden path.

INTEGRATION: this guide needs the prior's proj_logits to run its forward, so unlike
LogitGuide (which only sees h), it must be given a reference to proj_logits. In
gflow.py, build it with the frozen prior's projection:

    self.guide = HiddenGuide(d_model, proj_logits=self.frozen.proj_logits,
                             hidden=cfg.guide_hidden, layers=cfg.guide_layers)

and everywhere you form guided logits:
    guided = self.guide.guided_logits(h)          # note: takes h, returns LOGITS
(the guide applies proj_logits internally, so the call site should NOT add
proj_logits again -- see the wiring note at the bottom.)
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


# ============================================================================
# WIRING NOTES for gflow.py
# ============================================================================
#
# The hidden guide is DIFFERENT from LogitGuide/TempGainGuide in one key way:
# it produces the FULL guided logits from h (it applies proj_logits itself),
# whereas the old guides produced a residual ADDED to a separately-computed
# proj_logits(h). So the call sites change shape.
#
# 1) BUILD (LitGFlowNet.__init__), passing the frozen projection:
#
#       from hidden_guide import HiddenGuide
#       self.guide = HiddenGuide(
#           d_model, proj_logits=self.frozen.proj_logits,
#           hidden=self.cfg.guide_hidden, layers=self.cfg.guide_layers,
#           vocab_size=self.cfg.vocab_size,
#           also_output_residual=self.cfg.hidden_guide_out_residual)
#
#    (self.frozen is the frozen Quetzal; its proj_logits is already no-grad.)
#
# 2) CALL SITES. Anywhere that currently does:
#
#       prior_logits = prior.proj_logits(h)
#       if guide is None:
#           guided = prior_logits
#       elif hasattr(guide, 'guided_logits'):
#           guided = guide.guided_logits(prior_logits, h)   # OLD 2-arg form
#       else:
#           guided = prior_logits + guide(h)
#
#    becomes (note HiddenGuide.guided_logits takes ONLY h):
#
#       if guide is None:
#           guided = prior.proj_logits(h)
#       elif isinstance(guide, HiddenGuide):
#           guided = guide.guided_logits(h)                 # applies proj itself
#       elif hasattr(guide, 'guided_logits'):
#           guided = guide.guided_logits(prior.proj_logits(h), h)  # TempGain 2-arg
#       else:
#           guided = prior.proj_logits(h) + guide(h)
#
#    Do this in BOTH _generate_guided and _db_rollout_states.
#
# 3) CONFIG flag:
#       hidden_guide_out_residual: bool = False
#
# 4) IDENTITY CHECK at init (delta zero-init => guided == prior):
#       with torch.no_grad():
#           h = torch.randn(4, d_model)
#           assert torch.allclose(self.guide.guided_logits(h),
#                                  self.frozen.proj_logits(h), atol=1e-5)
#
# 5) WARM START: the hidden guide's delta lives in a different space than the old
#    output-residual guide, so you CANNOT warm-start delta from an old LogitGuide.
#    Train it fresh (it starts at identity, so that's fine). If you set
#    also_output_residual=True, that output head COULD be warm-started from an old
#    guide, but keep it simple first.
#
# WHY THIS SHOULD BEAT THE CEILING (and how to verify):
#   proj_logits maps d_model -> 128. Its weight matrix W has rows w_a (one per
#   atom). logit_a = w_a . h. To raise atom a's logit past the winner, delta only
#   needs a component along (w_a - w_winner); because ||w_a|| is large, a SMALL
#   delta produces a LARGE logit change -- exactly the leverage the output-residual
#   guide lacked. After training, re-run ablate_logit_flip / probe: the flip rate
#   in the high-gap bins should now be > 0 (the ceiling test).