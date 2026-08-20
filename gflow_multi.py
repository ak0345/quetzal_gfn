"""
Compose two (or more) *already-trained* RTB guides at inference time, following
"Routing by Reaching" (Yoon et al., 2026), and compare the composed sampler
against each single guide and the base Quetzal via reward histograms + FCD.

No training happens here. We load N frozen guide checkpoints produced by gflow.py
(each a LogitGuide + a learned logZ, all sharing the same frozen Quetzal prior),
and build ONE shared trajectory whose next-atom distribution is the mixed policy.
------------------------------------------------------------------------------
THE MIXING RULE (DB F route, P_B = 1 because Quetzal is autoregressive with a
fixed atom order -> unique construction trajectory -> backward policy is 1).

For component i, along the shared trajectory the reaching probability is
    u_i(s_t) = prod_{j<=t} p_{i,F}(a_j | s_{j-1})            (Eq. 23 with P_B=1)
i.e. the running product of THAT guide's forward prob of the actions actually
taken by the *mixed* sampler. We keep it in log space: log_u_i += log p_{i,F}(a_t).

Z_i = exp(logZ_i) is read from each run's checkpoint (the learned RTB partition
function). The per-step, per-component weight is
    w_i(s_t) = omega_i * Z_i * u_i(s_t)            (in log: log omega_i + logZ_i + log_u_i)

Operators (G applied per step to the weighted forward dists W_i(.) := w_i * p_{i,F}(.)):
  * linear    : p_M ∝ sum_i W_i                         (EXACT for beta=1, Prop 4.1)
  * product ⊗ : p_M ∝ prod_i (W_i)^omega_i  (product-of-experts, "all high"; approx)
                (paper's harmonic-mean ⊗ is (p1 p2)/(p1+p2); we expose both, see
                 --product_kind {poe, harmonic})
  * contrast ◐: p_M ∝ W_1^2 / (W_1 + W_2)               ("first high, others low"; approx)
------------------------------------------------------------------------------
Each guide i was trained to sample p_i(x) ∝ p_prior(x) * R_i(x)^beta_i / Z_i,
i.e. a *tilted* distribution, NOT p_i(x) ∝ R_i(x). Two ways to compose:

  compose_space = "tilted"  (clean, default)
    Treat p_i := the tilted distribution the guide actually samples. Compose those
    directly. The shared Quetzal prior FACTORS OUT of every operator:
        linear  -> p_prior(x) * sum_i omega_i R_i(x)^beta_i
        product -> p_prior(x) * prod_i R_i(x)^(beta_i omega_i)
    Everything stays anchored to valid chemistry. Z_i = each run's learned logZ.
    This uses the plain per-step forward dists exactly as above.

  compose_space = "reward"  (Eq. 30/31 recovery)
    Target linear scalarization of the *base* rewards, p_M ∝ (sum_i omega_i R_i)^beta.
    Since ingredient models are tilted by beta_i, recover R_i = (Z_i p_i)^(1/beta_i)
    and mix with the reward-sharpened policy Eq. (31):
        p_M,F ∝ ( sum_i omega_i ( Z_i u_i(s) p_{i,F}(.|s) )^(1/beta_i) )^beta
    Set the outer sharpening beta with --compose_beta (default: mean of beta_i).
    This form is only derived for the linear operator; product and contrast
    still use the tilted-space operators above, and combining reward space with
    a non-linear operator warns.
------------------------------------------------------------------------------
Outputs: overlaid reward histograms (base / guide_i / composed), a 2D density
plot (reward_1 vs reward_2) showing base vs single-guides vs composed sitting in
the compromise region, per-reward summary stats, and optional FCD vs a reference.

------------------------------------------------------------------------------
DEFAULTS (this file):
  Composes the Osimertinib component guides, each trained on one leaf scorer of
  the Osimertinib MPO objective, then scores every sample on the full assembled
  objective the guides never saw, plus each component for the multi-objective
  diagnostics.
"""
import os
import json
import argparse
import importlib.util
from dataclasses import dataclass, asdict, fields
from types import SimpleNamespace

import numpy as np
import numpy as np, scipy
if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

torch.set_float32_matmul_precision("medium")

from model import Quetzal
from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from metrics import compute_valid_unique
from reward_fn import build_reward, mol_to_rdkit
from rdkit import Chem

# Reuse building blocks from the training script so behaviour matches exactly.
from tempgain_guide import TempGainGuide
from gflow import (
    LogitGuide,
    load_frozen_quetzal,
    mols_to_smiles,
    _get_fcd,
    _load_ref_smiles,
    LogFlowHead,
    GFNConfig,
)

# ============================ Configuration ================================
@dataclass
class MultiConfig:
    # --- prior (must match how the guides were trained) ---
    quetzal_ckpt: str = "geom.ckpt"
    train_module: str = "train.py"
    use_ema_prior: bool = True
    dataset: str = "geom"
    max_len: int = 192
    diff_steps: int = 18
    mask_atoms: str = None

    route: str = "policy"           # policy (Σ logP_F) | flow (logF + logP_F(STOP))
    flow_hidden: int = 512          # must match how the DB guides were trained
    flow_layers: int = 2

    # --- the trained guides to compose (comma-separated .ckpt paths) ---
    
    guide_ckpts: str = ("logs/quetzal-gfn/gfn-geom-osim-comp0--beta50/checkpoints/last.ckpt,"
                        "logs/quetzal-gfn/gfn-geom-osim-comp1--beta50/checkpoints/last.ckpt,"
                        "logs/quetzal-gfn/gfn-geom-osim-comp2--beta50/checkpoints/last.ckpt,"
                        "logs/quetzal-gfn/gfn-geom-osim-comp3--beta50/checkpoints/last.ckpt")
    # human labels for plots/summaries, comma-separated, same order as guide_ckpts
    guide_labels: str = "osim-c0-simFCFP4,osim-c1-simECFP6,osim-c2-tpsa,osim-c3-logp"
    # which guide weights to read from each ckpt: "ema" (eval-time) or "policy" (live)
    guide_source: str = "ema"
 
    # --- guide architecture (must match training) ---
    vocab_size: int = 128
    guide_hidden: int = 512
    guide_layers: int = 2
 
    # --- the rewards to *evaluate* each sample under (comma-separated specs) ---
    # Grammar (see _build_eval_rewards for the full list):
    #   qed | logp:<t>:<s> | tpsa:<t>:<s> | similarity:<smiles>:<thr> | isomer:<formula>
    #   guacamol:<bench_fn>            -> FULL assembled benchmark
    #   gcomp:<bench_key>:<idx|label>  -> ONE component of a benchmark
    # Append '=name' to force a display name.
    # Here: FULL Osimertinib MPO (primary), then the 4 components for diagnostics.
    eval_rewards: str = ("guacamol:hard_osimertinib=osim_MPO,"
                         "gcomp:osimertinib:0=c0_simFCFP4,"
                         "gcomp:osimertinib:1=c1_simECFP6,"
                         "gcomp:osimertinib:2=c2_tpsa,"
                         "gcomp:osimertinib:3=c3_logp")
    reward_beta: float = 1.0        # only used if an eval reward needs it; scoring uses base logr
    invalid_logr: float = -5.0
    force_method: str = "xtb"
 
    # --- composition ---
    operator: str = "product"       # linear | product | contrast
    product_kind: str = "poe"       # poe (prod of experts) | harmonic (paper's ⊗)
    weights: str = "0.25,0.25,0.25,0.25"   # omega_i, comma-separated, same order as guides
    compose_space: str = "tilted"   # tilted | reward   (see module docstring)
    train_betas: str = "50,50,50,50"       # beta_i each guide was TRAINED with
    compose_beta: float = 0.0       # outer sharpening beta for reward-space linear; 0 => mean(train_betas)
    use_logz: bool = False          # weight components by learned Z_i; if False, Z_i := 1
    logz_override: str = ""         # optional comma-separated logZ values to override checkpoint
    # --- sampling / eval ---
    n_samples: int = 2000
    chunk: int = 500
    sample_temp: float = 1.0
    rand_eps: float = 0.0
    seed: int = 0
    device: str = "cuda"

    # --- outputs ---
    out_dir: str = "logs/quetzal-gfn/osim-compose/compose"
    tag: str = "peri_compose_db"
    make_base: bool = False          # also sample pure Quetzal
    make_singles: bool = False       # also sample each single guide alone
    fcd_ref_smiles: str = None
    fcd_enabled: bool = False
    save_mols: bool = True
    hv_ref: float = 0.0             # reference point per objective for hypervolume (anti-ideal)

    # --- plotting window ---
    plot_clip: str = "auto"         # "auto" (percentile), "none", or "lo,hi" e.g. "-6,0"
    plot_pct: float = 2.0           # lower-percentile clip when plot_clip="auto" (drops invalid tail)
    drop_invalid_plots: bool = True # exclude logr<=invalid_logr from density/KDE (kept in stats)
    ternary_norm: str = "minmax"    # none | minmax | rank  (per-objective normalization for ternary)

# ============================ Guide loading ================================
def _read_state_dict(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["state_dict"] if "state_dict" in ckpt else ckpt

def _extract_guide(sd, d_model, cfg: MultiConfig, source="ema", proj_logits=None):
    """Rebuild a guide from a LitGFlowNet checkpoint state_dict.

    Detects the guide TYPE from the checkpoint keys:
      * HiddenGuide: keys 'guide[_ema.module].delta.*' / '._proj.*', rebuilt as
        HiddenGuide(d_model, proj_logits, ...). Requires proj_logits.
      * LogitGuide, optionally wrapped in a TempGainGuide: 'net.*' keys.
    """
    if source == "ema":
        base_prefix = "guide_ema.module."
    elif source == "policy":
        base_prefix = "guide."
    else:
        raise ValueError(f"guide_source must be 'ema' or 'policy', got {source!r}")

    def strip(prefix):
        return {k[len(prefix):]: v for k, v in sd.items()
                if k.startswith(prefix) and "n_averaged" not in k}

    # ---- detect HiddenGuide by its signature keys ('delta.' / 'out_residual.') ----
    # HiddenGuide registers self._proj = proj_logits (an nn.Linear), so the ckpt
    # ALSO contains '<prefix>_proj.weight/bias' = the FROZEN prior's projection
    # weights (duplicated). Detection keys off the TRAINED modules delta/out_residual.
    ema_sub = strip(base_prefix)
    is_hidden = any(kk.startswith(("delta.", "out_residual."))
                    for kk in ema_sub)
    if is_hidden:
        from hidden_guide import HiddenGuide
        if proj_logits is None:
            raise RuntimeError(
                "HiddenGuide checkpoint detected but proj_logits was not passed to "
                "_extract_guide. Update Composer.__init__ to pass "
                "proj_logits=self.prior.proj_logits (see patch Change 1).")
        has_out_res = any(kk.startswith("out_residual.") for kk in ema_sub)
        # Hidden WIDTH from delta.0.weight [hidden, d_model] -- authoritative.
        # LAYER COUNT from the number of Linear entries in delta: HiddenGuide builds
        # delta as (layers-1)x[Linear,SiLU] + 1 Linear, so #linears == layers exactly
        # (verified: layers=1->{delta.0}; layers=2->{delta.0,delta.2}; etc).
        vs = getattr(cfg, "vocab_size", 128)
        gh = int(ema_sub["delta.0.weight"].shape[0]) if "delta.0.weight" in ema_sub \
             else getattr(cfg, "guide_hidden", 512)
        n_linears = len([k for k in ema_sub
                         if k.startswith("delta.") and k.endswith(".weight")])
        gl_ = n_linears if n_linears > 0 else getattr(cfg, "guide_layers", 2)
        guide = HiddenGuide(
            d_model, proj_logits=proj_logits,
            hidden=gh, layers=gl_, vocab_size=vs, also_output_residual=has_out_res)
        # _proj is already set to the passed (frozen) proj_logits by the ctor; the
        # ckpt's _proj.* keys are the SAME frozen weights, so strict=False load is
        # a harmless no-op match. Only delta/out_residual carry trained signal.
        missing, unexpected = guide.load_state_dict(ema_sub, strict=False)
        real_missing = [m for m in missing if "n_averaged" not in m]
        # Only delta/out_residual are TRAINED and must load. _proj.* are the frozen
        # prior's projection weights (registered because self._proj = proj_logits is
        # an nn.Module); they're identical to what the ctor already set, so whether
        # they load or not is immaterial. Guard on the trained modules only.
        trained_missing = [m for m in real_missing
                           if m.startswith(("delta.", "out_residual."))]
        if trained_missing:
            raise RuntimeError(
                f"HiddenGuide: missing trained params {trained_missing[:6]} for "
                f"source={source!r}. Checkpoint prefix/source likely wrong.")
        guide.eval()
        for p in guide.parameters():
            p.requires_grad = False
        with torch.no_grad():
            # weight-norm over the TRAINED params only (delta + out_residual),
            # excluding the frozen proj_logits handle.
            wnorm = sum(p.float().norm().item()
                        for n, p in guide.named_parameters()
                        if n.startswith(("delta", "out_residual")))
        print(f"[_extract_guide] HiddenGuide source={source} "
              f"trained_wnorm={wnorm:.3f} out_residual={has_out_res}")
        return guide, wnorm

    # ---------- original LogitGuide / TempGainGuide path (unchanged) ----------
    guide = LogitGuide(d_model, cfg.vocab_size, cfg.guide_hidden, cfg.guide_layers)
    if source == "ema":
        base_prefix = "guide_ema.module."
    elif source == "policy":
        base_prefix = "guide."
    else:
        raise ValueError(f"guide_source must be 'ema' or 'policy', got {source!r}")

    def strip(prefix):
        return {k[len(prefix):]: v for k, v in sd.items()
                if k.startswith(prefix) and "n_averaged" not in k}

    # attempt 1: a bare LogitGuide, whose params sit directly under the prefix
    sub = strip(base_prefix)
    # No net.* params means a TempGainGuide, whose base guide is nested one
    # level deeper under base_prefix + 'guide.'
    if not any(kk.startswith("net.") for kk in sub):
        nested = base_prefix + "guide."
        sub_nested = strip(nested)
        if any(kk.startswith("net.") for kk in sub_nested):
            sub = sub_nested
            base_prefix = nested   # so temp/gain prefixes below line up

    missing, unexpected = guide.load_state_dict(sub, strict=False)
    real_missing = [m for m in missing if "n_averaged" not in m]
    if real_missing:
        # give an actionable error: show what prefixes DO exist in the ckpt
        sample_keys = sorted({k.split(".")[0] + "." + (k.split(".")[1] if "." in k[len(k.split(".")[0])+1:] else "")
                              for k in list(sd.keys())})[:12]
        raise RuntimeError(
            f"Missing guide params for {source!r}: {real_missing[:6]} ...\n"
            f"  tried base_prefix={base_prefix!r}; sliced {len(sub)} keys.\n"
            f"  checkpoint top-level key groups: {sample_keys}\n"
            f"  -> if you see 'guide_ema.module.guide.*' the ckpt is post-patch "
            f"(handled); if you see neither, the source/prefix is wrong.")
    guide.eval()
    for p in guide.parameters():
        p.requires_grad = False

    # wrap with TempGainGuide and load temp/gain heads if the ckpt has them.
    # temp/gain sit as SIBLINGS of the base under the TempGainGuide, i.e. at
    # <module_prefix>.temp.* and <module_prefix>.gain.* where module_prefix is the
    # TempGainGuide (base_prefix WITHOUT the trailing 'guide.').
    tg = TempGainGuide(guide, d_model, hidden=getattr(cfg, 'tempgain_hidden', 128),
                       use_temperature=True, use_gain=True)
    module_prefix = base_prefix[:-len("guide.")] if base_prefix.endswith("guide.") \
        else base_prefix
    tp, gp = module_prefix + "temp.", module_prefix + "gain."
    tsub = strip(tp); gsub = strip(gp)
    loaded_heads = []
    if tsub and tg.temp is not None:
        tg.temp.load_state_dict(tsub, strict=False); loaded_heads.append("temp")
    if gsub and tg.gain is not None:
        tg.gain.load_state_dict(gsub, strict=False); loaded_heads.append("gain")
    tg.eval()
    for p in tg.parameters():
        p.requires_grad = False
    with torch.no_grad():
        wnorm = sum(p.float().norm().item() for p in guide.parameters())
    print(f"[_extract_guide] source={source} base_wnorm={wnorm:.3f} "
          f"heads_loaded={loaded_heads or 'none (identity)'}")
    return tg, wnorm

def _extract_flow_head(sd, d_model, cfg, source="ema"):
    """Rebuild the LogFlowHead from a DB checkpoint. Returns None if absent
    (i.e. the guide was trained with RTB, not DB)."""
    # flow head is stored under 'flow_head.*' (policy) and has no EMA by default
    prefix = "flow_head."
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        return None
    head = LogFlowHead(d_model, cfg.flow_hidden, cfg.flow_layers)
    missing, _ = head.load_state_dict(sub, strict=False)
    real_missing = [m for m in missing if "n_averaged" not in m]
    if real_missing:
        raise RuntimeError(f"Missing flow_head params: {real_missing[:6]} ...")
    head.eval()
    for p in head.parameters():
        p.requires_grad = False
    return head

def _extract_logz(sd):
    if "logZ" in sd:
        return float(sd["logZ"])
    for k, v in sd.items():
        if k.endswith("logZ"):
            return float(v)
    return 0.0

# ============================ Composed generation ==========================
@torch.no_grad()
def generate_composed(prior, guides, log_weights, log_Z, train_betas,
                      operator, product_kind, compose_space, compose_beta,
                      bsz, max_len, device, mask_atoms=None,
                      sample_temp=1.0, rand_eps=0.0, pbar=False,
                      flow_heads=None, route="policy", **coord_kwargs):
    """
    Roll out ONE shared trajectory whose next-atom distribution is the composed
    mixing policy over `guides`. Returns Molecule batch + accumulated per-component
    log-reaching-probabilities (diagnostics).
    """
    k = len(guides)
    if mask_atoms is None or mask_atoms == "None":
        mask = torch.ones(128, dtype=torch.bool, device=device)
        mask[GEN] = False
        mask[PAD] = False
    elif mask_atoms == "qm9":
        mask = QM9_MASK.to(device)
    elif mask_atoms == "H":
        mask = torch.zeros(128, dtype=torch.bool, device=device)
        mask[0] = True
        mask[1] = True
    elif isinstance(mask_atoms, list):
        mask = torch.zeros(128, dtype=torch.bool, device=device)
        mask[mask_atoms] = True
    else:
        raise ValueError(f"Unknown mask_atoms {mask_atoms!r}")

    max_len = max_len or prior.block_size
    NEG = -1e9
    uniform = mask.float()
    uniform = uniform / uniform.sum()

    atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
    coords = torch.zeros(bsz, 1, 3, device=device)
    stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)

    log_u = torch.zeros(bsz, k, device=device)
    log_p_composed = torch.zeros(bsz, device=device)   # Σ_t log P_M(a_t | s_t)
    log_weights = log_weights.to(device)
    log_Z = log_Z.to(device)
    inv_beta = (1.0 / train_betas.to(device)).view(1, k, 1)

    rng = tqdm.trange(max_len) if pbar else range(max_len)
    for _ in rng:
        idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)
        seq = prior.encode1(idx, atoms, coords)
        h = seq[:, -1, :]
        prior_logits = prior.proj_logits(h)

        logp = []
        for gi in guides:
            if gi is None:
                gl = prior_logits
            elif hasattr(gi, 'guided_logits'):
                try:
                    gl = gi.guided_logits(prior_logits, h)   # LogitGuide/TempGain (2-arg)
                except TypeError:
                    gl = gi.guided_logits(h)                 # HiddenGuide (1-arg)
            else:
                gl = prior_logits + gi(h)
            gl = gl.float().masked_fill(~mask, NEG)
            logp.append(F.log_softmax(gl, dim=-1))
        logp = torch.stack(logp, dim=1)  # [bsz, k, V]

        if route == "flow":
            # reconstructed per-component log-reward at THIS state:
            #   log R_i(s) = log F_i(s) + log P_F^i(STOP | s)
            # F_i(s) = logprior_partial(s) + f_i(h_s); the shared logprior_partial
            # is identical across components (same frozen prior, same trajectory),
            # so it is an additive constant that CANCELS in every normalized
            # operator below -> we can drop it and use f_i(h_s) directly.
            # log P_F^i(STOP|s) is logp[:, i, STOP].
            f_state = []
            for gi, fh in enumerate(flow_heads):
                fh_val = torch.zeros(bsz, device=device) if fh is None else fh(h)
                f_state.append(fh_val)
            f_state = torch.stack(f_state, dim=1)                     # [bsz, k]
            logR_recon = f_state + logp[:, :, STOP]                   # [bsz, k]
            # this reconstructed log-reward plays the role that (logZ_i + log_u_i)
            # played in the policy route: it is the per-component, action-independent
            # log-weight. Fold omega in as before.
            logw = log_weights.view(1, k) + logR_recon                # [bsz, k]
            logW = logw.unsqueeze(-1) + logp                          # [bsz, k, V]
            logWZ = logR_recon.unsqueeze(-1) + logp                   # omega-free
        else:
            logw = (log_weights.view(1, k) + log_Z.view(1, k) + log_u)  # [bsz, k]
            logW = logw.unsqueeze(-1) + logp                                  # [bsz, k, V]
            logWZ = (log_Z.view(1, k) + log_u).unsqueeze(-1) + logp           # [bsz, k, V]

        if compose_space == "reward" and operator == "linear":
            inner = (inv_beta * logWZ)
            mix = torch.logsumexp(log_weights.view(1, k, 1) + inner, dim=1)
            comp = compose_beta * mix
            comp = comp.masked_fill(~mask.view(1, -1), NEG)
            comp_logp = F.log_softmax(comp, dim=-1)
        elif operator == "linear":
            comp = torch.logsumexp(logW, dim=1)
            comp_logp = comp - torch.logsumexp(comp, dim=-1, keepdim=True)
        elif operator == "product":
            if product_kind == "poe":
                w = log_weights.exp().view(1, k, 1)
                comp = (w * logWZ).sum(dim=1)
                comp = comp.masked_fill(~mask.view(1, -1), NEG)
                comp_logp = comp - torch.logsumexp(comp, dim=-1, keepdim=True)
            elif product_kind == "harmonic":
                lognum = logW.sum(dim=1)
                logden = torch.logsumexp(logW, dim=1)
                comp = lognum - logden
                comp = comp.masked_fill(~mask.view(1, -1), NEG)
                comp_logp = comp - torch.logsumexp(comp, dim=-1, keepdim=True)
            else:
                raise ValueError(f"product_kind {product_kind!r}")
        elif operator == "contrast":
            logW1 = logW[:, 0, :]
            if k == 1:
                comp = logW1
            else:
                logrest = torch.logsumexp(logW[:, 1:, :], dim=1)
                logden = torch.logsumexp(torch.stack([logW1, logrest], 0), dim=0)
                comp = 2.0 * logW1 - logden
            comp = comp.masked_fill(~mask.view(1, -1), NEG)
            comp_logp = comp - torch.logsumexp(comp, dim=-1, keepdim=True)
        else:
            raise ValueError(f"operator {operator!r}")

        behav = F.softmax(comp_logp / sample_temp, dim=-1)
        if rand_eps > 0:
            behav = (1 - rand_eps) * behav + rand_eps * uniform
        next_atom = torch.multinomial(behav, num_samples=1)

        alive = (~stop_mask).float().unsqueeze(-1)
        chosen_logp = logp.gather(-1, next_atom.unsqueeze(1).expand(-1, k, -1)).squeeze(-1)
        log_u = log_u + chosen_logp * alive
        # composed-policy log-prob of the taken action, for the slope test:
        alive_1d = (~stop_mask).float()
        log_p_composed = log_p_composed + \
            comp_logp.gather(-1, next_atom).squeeze(-1) * alive_1d

        stop_mask = stop_mask | (next_atom.squeeze(-1) == STOP)
        if stop_mask.all():
            break
        atoms = torch.cat([atoms, next_atom], dim=1)
        x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
        next_coord, _ = prior.sample_coord(x, device=device, **coord_kwargs)
        coords = torch.cat([coords, next_coord.view(bsz, 1, 3)], dim=1)

    mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu")
    return mols, {"log_u": log_u.cpu(), "log_p_composed": log_p_composed.cpu()}

# ============================ Sampling driver ==============================
class Composer:
    def __init__(self, cfg: MultiConfig):
        self.cfg = cfg
        # guard: diff_steps=None (from a stale checkpoint/config) crashes
        # sample_coord; a degenerate low value produces garbage molecules.
        if getattr(self.cfg, "diff_steps", None) is None or self.cfg.diff_steps < 1:
            print(f"[cfg] diff_steps was {self.cfg.diff_steps!r}; defaulting to 18")
            self.cfg.diff_steps = 18
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        gfn_cfg = GFNConfig(
            quetzal_ckpt=cfg.quetzal_ckpt, train_module=cfg.train_module,
            use_ema_prior=cfg.use_ema_prior,
        )
        self.prior, prior_cfg = load_frozen_quetzal(gfn_cfg)
        self.prior.to(self.device).eval()
        self.d_model = prior_cfg.n_embd

        self.ckpts = [p for p in cfg.guide_ckpts.split(",") if p.strip()]
        self.labels = [s.strip() for s in cfg.guide_labels.split(",") if s.strip()]
        assert len(self.ckpts) >= 1, "provide --guide_ckpts"
        if len(self.labels) != len(self.ckpts):
            self.labels = [f"guide{i}" for i in range(len(self.ckpts))]

        self.guides, self.logZ, self.flow_heads = [], [], []
        for path, lab in zip(self.ckpts, self.labels):
            sd = _read_state_dict(path)
            g, wnorm = _extract_guide(sd, self.d_model, cfg, source=cfg.guide_source,
                                      proj_logits=self.prior.proj_logits)
            g.to(self.device)
            z = _extract_logz(sd)
            fh = _extract_flow_head(sd, self.d_model, cfg, source=cfg.guide_source)
            if fh is not None:
                fh.to(self.device)
            print(f"[load] {lab}: guide weight-norm={wnorm:.3f}  logZ={z:.3f}  "
                  f"flow_head={'yes' if fh is not None else 'NO'}  ({path})")
            if wnorm < 1e-6:
                print(f"[warn] {lab} guide weight-norm ~0 -> untrained/zero guide.")
            self.guides.append(g)
            self.logZ.append(z)
            self.flow_heads.append(fh)
        k = len(self.guides)

        if cfg.route == "flow":
            missing_fh = [self.labels[i] for i, fh in enumerate(self.flow_heads) if fh is None]
            if missing_fh:
                raise ValueError(
                    f"--route flow requires DB guides with a flow_head, but these "
                    f"have none (trained with RTB?): {missing_fh}. Re-train them with "
                    f"--objective db, or use --route policy.")

        if cfg.logz_override.strip():
            ov = [float(x) for x in cfg.logz_override.split(",")]
            assert len(ov) == k
            self.logZ = ov
            print(f"[load] logZ overridden -> {self.logZ}")
        self.log_Z = torch.tensor(self.logZ, dtype=torch.float32)
        if not cfg.use_logz:
            self.log_Z = torch.zeros(k)

        w = [float(x) for x in cfg.weights.split(",")]
        assert len(w) == k, f"weights ({len(w)}) must match guides ({k})"
        w = np.asarray(w, dtype=float)
        self.log_weights = torch.tensor(np.log(np.clip(w, 1e-12, None)), dtype=torch.float32)

        tb = [float(x) for x in cfg.train_betas.split(",")]
        assert len(tb) == k
        self.train_betas = torch.tensor(tb, dtype=torch.float32)
        self.compose_beta = cfg.compose_beta if cfg.compose_beta > 0 else float(np.mean(tb))

        if cfg.compose_space == "reward" and cfg.operator != "linear":
            print(f"[warn] compose_space=reward is only derived for linear; "
                  f"operator={cfg.operator} will use tilted-space operator instead.")

        self.eval_rewards = self._build_eval_rewards()

    def _build_eval_rewards(self):
        """Parse cfg.eval_rewards into [(display_name, log_reward_fn), ...].

        Compatible with the current reward_fn: builds each scorer by handing
        build_reward a lightweight SimpleNamespace carrying exactly the fields
        that reward kind reads (no dependency on GFNConfig fields, which don't
        include the new guacamol_component selectors).

        Spec grammar (comma-separated), each spec colon-delimited; optional
        '=display_name' suffix:
            qed
            logp:<target>:<sigma>
            tpsa:<target>:<sigma>
            similarity:<smiles>:<threshold>
            isomer:<formula>
            guacamol:<bench_fn>              (FULL assembled benchmark)
            gcomp:<bench_key>:<idx|label>    (ONE component of a benchmark)
        """
        floor = self.cfg.invalid_logr
        specs = [s for s in self.cfg.eval_rewards.split(",") if s.strip()]
        fns, names = [], []
        for spec in specs:
            spec = spec.strip()
            disp = None
            if "=" in spec:
                spec, disp = spec.rsplit("=", 1)
                spec, disp = spec.strip(), disp.strip()
            parts = [p.strip() for p in spec.split(":")]
            kind = parts[0]

            def ns(**kw):
                base = dict(invalid_logr=floor, force_method=self.cfg.force_method,
                            dataset=self.cfg.dataset, reward_beta=1.0)
                base.update(kw)
                return SimpleNamespace(**base)

            if kind == "qed":
                rc = ns(reward="qed")
                default_name = "qed"
            elif kind == "logp":
                t = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                s = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
                rc = ns(reward="logp", reward_target=t, reward_sigma=s)
                default_name = f"logp(t={t},s={s})"
            elif kind == "tpsa":
                t = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                s = float(parts[2]) if len(parts) > 2 and parts[2] else 1.0
                rc = ns(reward="tpsa", reward_target=t, reward_sigma=s)
                default_name = f"tpsa(t={t},s={s})"
            elif kind == "similarity":
                smi = parts[1] if len(parts) > 1 else ""
                thr = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
                rc = ns(reward="similarity", reward_smiles=smi, reward_target=thr)
                default_name = "similarity"
            elif kind == "isomer":
                formula = parts[1] if len(parts) > 1 else ""
                rc = ns(reward="isomer", reward_formula=formula)
                default_name = f"isomer({formula})"
            elif kind == "guacamol":
                bench_fn = parts[1] if len(parts) > 1 else ""
                rc = ns(reward="guacamol", reward_smiles=bench_fn)
                default_name = bench_fn or "guacamol"
            elif kind == "gcomp":
                bench_key = parts[1] if len(parts) > 1 else ""
                comp = parts[2] if len(parts) > 2 else "0"
                # integer index if it parses as one, else a label substring
                try:
                    comp_sel = int(comp)
                except ValueError:
                    comp_sel = comp
                rc = ns(reward="guacamol_component",
                        reward_benchmark=bench_key, reward_component=comp_sel)
                default_name = f"{bench_key}:c{comp}"
            else:
                raise ValueError(f"Unknown eval reward kind {kind!r} in spec {spec!r}")

            fns.append(build_reward(rc))
            names.append(disp or default_name)
        print(f"[eval] scoring rewards: {names}")
        return list(zip(names, fns))

    @torch.no_grad()
    def sample_composed(self, n):
        return self._chunked(lambda b: generate_composed(
            self.prior, self.guides, self.log_weights, self.log_Z, self.train_betas,
            self.cfg.operator, self.cfg.product_kind, self.cfg.compose_space,
            self.compose_beta, b, self.cfg.max_len, self.device,
            mask_atoms=self.cfg.mask_atoms, sample_temp=self.cfg.sample_temp,
            rand_eps=self.cfg.rand_eps, flow_heads=self.flow_heads,
            route=self.cfg.route, num_steps=self.cfg.diff_steps,
        )[0], n)

    @torch.no_grad()
    def sample_composed_with_logp(self, n):
        """Like sample_composed but also returns model log P(x) per molecule
        under the composed policy (eval policy: temp=1, eps=0)."""
        mols_all, logp_all = [], []
        done = 0
        while done < n:
            b = min(self.cfg.chunk, n - done)
            mols, info = generate_composed(
                self.prior, self.guides, self.log_weights, self.log_Z,
                self.train_betas, self.cfg.operator, self.cfg.product_kind,
                self.cfg.compose_space, self.compose_beta, b, self.cfg.max_len,
                self.device, mask_atoms=self.cfg.mask_atoms,
                sample_temp=1.0, rand_eps=0.0,
                flow_heads=self.flow_heads, route=self.cfg.route,
                num_steps=self.cfg.diff_steps)
            mols_all.extend(mols.unbatch())
            logp_all.append(info["log_p_composed"])
            done += b
        return mols_all, torch.cat(logp_all).numpy()

    @torch.no_grad()
    def sample_single(self, gi, n):
        return self._chunked(lambda b: generate_composed(
            self.prior, [self.guides[gi]], torch.zeros(1), self.log_Z[gi:gi+1],
            self.train_betas[gi:gi+1], "linear", "poe", "tilted", 1.0,
            b, self.cfg.max_len, self.device, mask_atoms=self.cfg.mask_atoms,
            sample_temp=self.cfg.sample_temp, rand_eps=self.cfg.rand_eps,
            num_steps=self.cfg.diff_steps,
        )[0], n)

    @torch.no_grad()
    def sample_base(self, n):
        return self._chunked(lambda b: generate_composed(
            self.prior, [None], torch.zeros(1), torch.zeros(1), torch.ones(1),
            "linear", "poe", "tilted", 1.0, b, self.cfg.max_len, self.device,
            mask_atoms=self.cfg.mask_atoms, sample_temp=self.cfg.sample_temp,
            rand_eps=self.cfg.rand_eps, num_steps=self.cfg.diff_steps,
        )[0], n)

    def _chunked(self, fn, n):
        mols = []
        done = 0
        while done < n:
            b = min(self.cfg.chunk, n - done)
            m = fn(b).unbatch()
            mols.extend(m)
            done += b
        return mols

    def score(self, mols):
        out = {}
        for name, fn in self.eval_rewards:
            out[name] = np.asarray([fn(m) for m in mols], dtype=float)
        return out

# ============================ Reporting ====================================
def summary_stats(logr_dict, valid, unique):
    s = {"validity": valid, "uniqueness": unique}
    for name, arr in logr_dict.items():
        s[f"{name}/mean"] = float(np.mean(arr))
        s[f"{name}/median"] = float(np.median(arr))
        for k in (1, 10, 100):
            kk = min(k, len(arr))
            top = np.sort(arr)[::-1][:kk]
            s[f"{name}/top{k}"] = float(top.mean()) if kk else float("nan")
    return s

def _clip_window(all_vals, cfg):
    if cfg.plot_clip == "none":
        return float(np.min(all_vals)), float(np.max(all_vals))
    if "," in cfg.plot_clip:
        lo, hi = cfg.plot_clip.split(",")
        return float(lo), float(hi)
    lo = float(np.percentile(all_vals, cfg.plot_pct))
    hi = float(np.max(all_vals))
    return lo, min(hi, 0.0) if hi <= 0.5 else hi

def _prep(arr, cfg, lo=None):
    a = np.asarray(arr, dtype=float)
    if cfg.drop_invalid_plots:
        a = a[a > cfg.invalid_logr + 1e-6]
    if lo is not None:
        a = a[a >= lo]
    return a

def plot_hist(series, reward_name, out_path, title, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allv = np.concatenate([_prep(v, cfg) for v in series.values() if len(v)])
    if len(allv) == 0:
        return
    lo, hi = _clip_window(allv, cfg)
    bins = np.linspace(lo, hi, 45)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    for label, arr in series.items():
        a = _prep(arr, cfg, lo=lo)
        a = a[a <= hi]
        if len(a) == 0:
            continue
        ax1.hist(a, bins=bins, density=True, histtype="step", linewidth=2, label=label)
        xs = np.sort(a)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, ys, linewidth=2, label=label)
    ax1.set_ylabel("density")
    ax1.set_title(title)
    ax1.legend()
    ax2.set_ylabel("cumulative frac")
    ax2.set_xlabel(f"log reward [{reward_name}]")
    ax2.set_xlim(lo, hi)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def plot_joint(points, name_x, name_y, out_path, title, cfg=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for label, (xs, ys) in points.items():
        if len(xs):
            ax.scatter(xs, ys, s=10, alpha=0.35, label=label)
    ax.set_xlabel(f"log reward [{name_x}]")
    ax.set_ylabel(f"log reward [{name_y}]")
    if cfg is not None:
        allx = np.concatenate([x for x, _ in points.values() if len(x)])
        ally = np.concatenate([y for _, y in points.values() if len(y)])
        lox, hix = _clip_window(allx, cfg)
        loy, hiy = _clip_window(ally, cfg)
        ax.set_xlim(lox, hix)
        ax.set_ylim(loy, hiy)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def _pareto_front(objs):
    n = objs.shape[0]
    dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if dominated[i]:
            continue
        ge = np.all(objs >= objs[i], axis=1)
        gt = np.any(objs > objs[i], axis=1)
        if np.any(ge & gt):
            dominated[i] = True
    return ~dominated

def hypervolume(objs, ref):
    objs = np.asarray(objs, dtype=float)
    ref = np.asarray(ref, dtype=float)
    keep = np.all(objs > ref, axis=1)
    objs = objs[keep]
    if len(objs) == 0:
        return 0.0
    front = objs[_pareto_front(objs)]
    m = front.shape[1]
    if m == 2:
        order = np.argsort(-front[:, 0])
        f = front[order]
        hv = 0.0
        prev_y = ref[1]
        for x, y in f:
            if y <= prev_y:
                continue
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
        return float(hv)
    hi = front.max(axis=0)
    box = np.prod(hi - ref)
    if box <= 0:
        return 0.0
    N = 200_000
    pts = ref + np.random.rand(N, m) * (hi - ref)
    dominated = np.zeros(N, dtype=bool)
    for p in front:
        dominated |= np.all(pts <= p, axis=1)
    return float(box * dominated.mean())

def logr_to_score(arr, invalid_logr, score_floor=None):
    """Map log reward -> underlying [0,1] score via exp.

    Two distinct floors, because "genuine invalid" and "valid-but-low" are
    different things and a single global cutoff mislabels one as the other when
    eval objectives have different valid ranges:

      * invalid_logr : the TRUE-invalid cutoff. Log-rewards at/near this value
        are parse/scoring failures and map to exactly 0. This is the eval
        reward's own floor (cfg.invalid_logr), shared across objectives.
      * score_floor  : the lower end of this objective's VALID range, used only
        to clip huge-negative valid log-rewards before exp() so they don't
        underflow to 0 and swamp other objectives. Defaults to invalid_logr for
        backward-compatible behaviour; pass a per-objective value (e.g. just
        below the objective's min finite log-reward) for clean cross-objective
        score normalization.

    A value is forced to score 0 iff it is a genuine invalid (<= invalid_logr +
    0.1). Valid values below score_floor are clipped to score_floor (kept > 0),
    NOT zeroed.
    """
    a = np.asarray(arr, dtype=float)
    if score_floor is None:
        score_floor = invalid_logr
    # Genuine invalids are detected by EXACT equality to the eval floor: a failed
    # RDKit conversion / scoring returns precisely cfg.invalid_logr, whereas a
    # valid molecule's score is continuous and never lands exactly on the floor.
    # This is what lets a VALID molecule whose log-reward happens to sit below the
    # floor (e.g. a TPSA/logP Gaussian tail at -6.6 when the floor is -5) be kept
    # as a small positive score instead of being mislabeled invalid.
    invalid = np.isclose(a, invalid_logr, atol=1e-6)
    a = np.clip(a, score_floor, 0.0)
    s = np.exp(a)
    s[invalid] = 0.0
    return np.clip(s, 0.0, 1.0)


def compute_score_floors(scored, reward_names, cfg):
    """Per-objective score_floor for logr_to_score, derived from the pooled
    finite log-rewards across ALL runs (so cross-run comparison is preserved).

    For each objective, floor = (min valid log-reward) - margin. "Valid" means
    NOT a genuine invalid, where genuine invalids are detected by exact equality
    to cfg.invalid_logr (a failed conversion returns precisely the floor; a real
    score never lands exactly on it). This correctly INCLUDES valid molecules
    whose log-reward dips below cfg.invalid_logr (e.g. a Gaussian TPSA tail at
    -6.6 with floor -5), so their score is preserved rather than zeroed.
    Falls back to cfg.invalid_logr if an objective has no valid values.
    """
    margin = 0.5
    floors = {}
    for rn in reward_names:
        vals = []
        for label in scored:
            a = np.asarray(scored[label][rn], dtype=float)
            a = a[~np.isclose(a, cfg.invalid_logr, atol=1e-6)]   # drop genuine invalids only
            if len(a):
                vals.append(a)
        if vals:
            mn = float(np.min(np.concatenate(vals)))
            # sit a touch below the worst valid value; never above 0.
            floors[rn] = min(mn - margin, 0.0)
        else:
            floors[rn] = cfg.invalid_logr
    return floors

def plot_kde(points, name_x, name_y, out_path, title, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        return plot_joint(points, name_x, name_y, out_path, title, cfg)
    prepped = {}
    for label, (xs, ys) in points.items():
        xs = np.asarray(xs, float); ys = np.asarray(ys, float)
        if cfg.drop_invalid_plots:
            keep = (xs > cfg.invalid_logr + 1e-6) & (ys > cfg.invalid_logr + 1e-6)
            xs, ys = xs[keep], ys[keep]
        prepped[label] = (xs, ys)
    allx = np.concatenate([x for x, _ in prepped.values() if len(x)])
    ally = np.concatenate([y for _, y in prepped.values() if len(y)])
    if len(allx) < 5:
        return plot_joint(points, name_x, name_y, out_path, title, cfg)
    xlo, xhi = _clip_window(allx, cfg)
    ylo, yhi = _clip_window(ally, cfg)
    xx, yy = np.mgrid[xlo:xhi:120j, ylo:yhi:120j]
    grid = np.vstack([xx.ravel(), yy.ravel()])
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#555555"]
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for i, (label, (xs, ys)) in enumerate(prepped.items()):
        c = color_cycle[i % len(color_cycle)]
        win = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
        xs, ys = xs[win], ys[win]
        if len(xs) < 15 or np.std(xs) < 1e-4 or np.std(ys) < 1e-4:
            ax.scatter(xs, ys, s=10, alpha=0.4, color=c, label=label)
            continue
        try:
            kde = gaussian_kde(np.vstack([xs, ys]))
            dens = kde(grid).reshape(xx.shape)
            dmax = dens.max()
            levels = dmax * np.array([0.1, 0.3, 0.5, 0.7, 0.9])
            ax.contour(xx, yy, dens, levels=levels, colors=[c], linewidths=1.4, alpha=0.9)
            ax.plot([], [], color=c, lw=2.2, label=label)
        except Exception:
            ax.scatter(xs, ys, s=10, alpha=0.4, color=c, label=label)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"log reward [{name_x}]")
    ax.set_ylabel(f"log reward [{name_y}]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def plot_kde_grid(scored, reward_names, out_path, title, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from scipy.stats import gaussian_kde
    except Exception:
        return
    m = len(reward_names)
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
    ncol = min(3, len(pairs))
    nrow = int(np.ceil(len(pairs) / ncol))
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#555555"]
    labels = list(scored.keys())
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow), squeeze=False)
    for p_idx, (i, j) in enumerate(pairs):
        ax = axes[p_idx // ncol][p_idx % ncol]
        nx, ny = reward_names[i], reward_names[j]
        px, py = [], []
        prepped = {}
        for label in labels:
            xs = np.asarray(scored[label][nx], float)
            ys = np.asarray(scored[label][ny], float)
            if cfg.drop_invalid_plots:
                keep = (xs > cfg.invalid_logr + 1e-6) & (ys > cfg.invalid_logr + 1e-6)
                xs, ys = xs[keep], ys[keep]
            prepped[label] = (xs, ys)
            px.append(xs); py.append(ys)
        px = np.concatenate([a for a in px if len(a)]) if any(len(a) for a in px) else np.array([0.])
        py = np.concatenate([a for a in py if len(a)]) if any(len(a) for a in py) else np.array([0.])
        xlo, xhi = _clip_window(px, cfg)
        ylo, yhi = _clip_window(py, cfg)
        xx, yy = np.mgrid[xlo:xhi:100j, ylo:yhi:100j]
        grid = np.vstack([xx.ravel(), yy.ravel()])
        for li, label in enumerate(labels):
            c = color_cycle[li % len(color_cycle)]
            xs, ys = prepped[label]
            win = (xs >= xlo) & (xs <= xhi) & (ys >= ylo) & (ys <= yhi)
            xs, ys = xs[win], ys[win]
            if len(xs) < 15 or np.std(xs) < 1e-4 or np.std(ys) < 1e-4:
                ax.scatter(xs, ys, s=6, alpha=0.3, color=c, label=label)
                continue
            try:
                kde = gaussian_kde(np.vstack([xs, ys]))
                dens = kde(grid).reshape(xx.shape)
                levels = dens.max() * np.array([0.1, 0.4, 0.7, 0.9])
                ax.contour(xx, yy, dens, levels=levels, colors=[c], linewidths=1.2, alpha=0.9)
                ax.plot([], [], color=c, lw=2.0, label=label)
            except Exception:
                ax.scatter(xs, ys, s=6, alpha=0.3, color=c, label=label)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel(f"logR [{nx}]"); ax.set_ylabel(f"logR [{ny}]")
        if p_idx == 0:
            ax.legend(fontsize=8)
    for p_idx in range(len(pairs), nrow * ncol):
        axes[p_idx // ncol][p_idx % ncol].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def plot_min_reward(scored, reward_names, out_path, title, cfg, score_floors=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sf = score_floors or {}
    series = {}
    for label in scored:
        cols = [logr_to_score(scored[label][rn], cfg.invalid_logr, sf.get(rn))
                for rn in reward_names]
        series[label] = np.min(np.stack(cols, axis=1), axis=1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    bins = np.linspace(0, 1, 41)
    for label, arr in series.items():
        if len(arr) == 0:
            continue
        ax1.hist(arr, bins=bins, density=True, histtype="step", linewidth=2, label=label)
        xs = np.sort(arr)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax2.plot(xs, ys, linewidth=2, label=label)
    ax1.set_ylabel("density"); ax1.set_title(title); ax1.legend()
    ax2.set_ylabel("cumulative frac")
    ax2.set_xlabel("min score across objectives (worst objective)")
    ax2.set_xlim(0, 1); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return {label: float(np.mean(arr)) for label, arr in series.items()}

def plot_ternary(scored, reward_names, out_path, title, cfg, score_floors=None):
    if len(reward_names) != 3:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sf = score_floors or {}
    norm = getattr(cfg, "ternary_norm", "rank")
    raw = {label: np.stack([logr_to_score(scored[label][rn], cfg.invalid_logr, sf.get(rn))
                            for rn in reward_names], axis=1)
           for label in scored}
    pooled = np.concatenate(list(raw.values()), axis=0)
    if norm == "minmax":
        lo = pooled.min(axis=0); hi = pooled.max(axis=0)
        rng = np.where(hi - lo > 1e-9, hi - lo, 1.0)
        def tf(s): return (s - lo) / rng
    elif norm == "rank":
        order = [np.sort(pooled[:, k]) for k in range(3)]
        def tf(s):
            out = np.empty_like(s)
            for k in range(3):
                out[:, k] = np.searchsorted(order[k], s[:, k], side="right") / len(order[k])
            return out
    else:
        def tf(s): return s
    V = np.array([[0.5, np.sqrt(3) / 2], [0.0, 0.0], [1.0, 0.0]])
    def to_xy(w): return w @ V
    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#555555"]
    fig, ax = plt.subplots(figsize=(6.6, 6.0))
    tri = np.vstack([V, V[0]])
    ax.plot(tri[:, 0], tri[:, 1], color="k", lw=1.0)
    cen = V.mean(0)
    ax.scatter(*cen, marker="+", s=120, color="k", zorder=5)
    for li, label in enumerate(scored):
        c = color_cycle[li % len(color_cycle)]
        s = tf(raw[label])
        tot = s.sum(axis=1)
        keep = tot > 1e-9
        s = s[keep] / tot[keep, None]
        xy = to_xy(s)
        ax.scatter(xy[:, 0], xy[:, 1], s=8, alpha=0.3, color=c, label=label)
    for v, name in zip(V, reward_names):
        off = (v - cen) * 0.10
        ax.annotate(name, v + off, ha="center", va="center", fontsize=10, fontweight="bold")
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{title}  [norm={norm}]")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser()
    for field in MultiConfig.__dataclass_fields__.values():
        if isinstance(field.default, bool):
            if field.default is False:
                parser.add_argument(f"--{field.name}", dest=field.name, action="store_true")
            else:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action="store_false")
            parser.set_defaults(**{field.name: field.default})
        else:
            argtype = type(field.default) if field.default is not None else str
            parser.add_argument(f"--{field.name}", type=argtype, default=field.default)
    return MultiConfig(**vars(parser.parse_args()))

def slope_test(log_p, log_r, invalid_logr=None, label="composed",
               weights_desc=None, out_png=None):
    """Regress model log P(x) on target log-reward. Slope==1 => correctly peaked.
    Correlation is NOT a substitute: an under-peaked sampler has high r, low slope."""
    log_p = np.asarray(log_p, dtype=float)
    log_r = np.asarray(log_r, dtype=float)
    m = np.isfinite(log_p) & np.isfinite(log_r)
    if invalid_logr is not None:
        m &= log_r > (invalid_logr + 0.1)
    x, y = log_r[m], log_p[m]
    n = len(x)
    if n < 3 or np.std(x) < 1e-9:
        return {"slope": float("nan"), "r": float("nan"), "n": n,
                "verdict": "too few points / degenerate"}
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    if r * r < 0.5:
        verdict = "DECOUPLED: low r, check upstream"
    elif slope > 0.85:
        verdict = "PASS: near-1, correctly peaked"
    elif slope > 0.6:
        verdict = "PARTIAL: somewhat under-peaked"
    else:
        verdict = "FAIL: under-peaked (ranks OK, mass wrong)"
    res = {"slope": float(slope), "intercept": float(intercept), "r": r,
           "r2": r * r, "n": int(n), "label": label,
           "weights_desc": weights_desc, "verdict": verdict}
    if out_png:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 6))
        ax.scatter(x, y, s=8, alpha=0.35, color="#4C72B0", label=f"{label} (n={n})")
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="#C44E52", lw=2,
                label=f"fit: slope={slope:.3f}, r={r:.3f}")
        x0, y0 = x.mean(), y.mean()
        ax.plot(xs, (xs - x0) + y0, "k--", lw=1.2, label="ideal slope=1")
        ax.set_xlabel("target log-reward  Σ w_k log R_k(x)")
        ax.set_ylabel("model log P(x)")
        ax.set_title(f"slope test  [{weights_desc}]" if weights_desc else "slope test")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_png, dpi=140)
        plt.close(fig)
    return res

def main():
    cfg = parse_args()
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(cfg)
    comp = Composer(cfg)
    k = len(comp.guides)

    runs = {}
    print(f"[sample] composed ({cfg.operator}) x {cfg.n_samples} ...")
    runs["composed"] = comp.sample_composed(cfg.n_samples)

    if cfg.make_singles:
        for i, lab in enumerate(comp.labels):
            print(f"[sample] single '{lab}' x {cfg.n_samples} ...")
            runs[lab] = comp.sample_single(i, cfg.n_samples)

    if cfg.make_base:
        print(f"[sample] base Quetzal x {cfg.n_samples} ...")
        runs["base"] = comp.sample_base(cfg.n_samples)

    all_summ = {}
    scored = {}
    for label, mols in runs.items():
        valid, unique = compute_valid_unique(mols)
        logr = comp.score(mols)
        scored[label] = logr
        all_summ[label] = summary_stats(logr, valid, unique)
        print(f"[score] {label}: "
              + " | ".join(f"{n}={np.mean(a):.3f}" for n, a in logr.items())
              + f" | valid={valid:.3f} uniq={unique:.3f}")

    reward_names = [n for n, _ in comp.eval_rewards]

    # Per-objective score floors, fit on the pooled finite log-rewards across all
    # runs. Used everywhere logr_to_score feeds a [0,1]-score-space metric/plot
    # (hypervolume, worst-objective, ternary) so objectives with different valid
    # ranges are normalized consistently and only GENUINE invalids map to 0.
    score_floors = compute_score_floors(scored, reward_names, cfg)
    print("[score] per-objective score floors: "
          + ", ".join(f"{rn}={score_floors[rn]:.2f}" for rn in reward_names))

    if len(reward_names) >= 2:
        ref = np.array([cfg.hv_ref] * len(reward_names), dtype=float)
        for label in runs:
            scores = np.stack(
                [logr_to_score(scored[label][rn], cfg.invalid_logr, score_floors.get(rn))
                 for rn in reward_names],
                axis=1)
            hv = hypervolume(scores, ref)
            n_pareto = int(_pareto_front(scores[np.all(scores > ref, axis=1)]).sum()) \
                if np.any(np.all(scores > ref, axis=1)) else 0
            all_summ[label]["hypervolume"] = hv
            all_summ[label]["pareto_count"] = n_pareto
            print(f"[hv] {label}: hypervolume={hv:.4f}  pareto_count={n_pareto}")

    for rn in reward_names:
        series = {label: scored[label][rn] for label in runs}
        out_png = os.path.join(cfg.out_dir, f"{cfg.tag}_hist_{rn}.png")
        plot_hist(series, rn, out_png,
                  f"{cfg.operator} composition — {rn}", cfg)
        print(f"[plot] {out_png}")

    if len(reward_names) >= 2:
        nx, ny = reward_names[0], reward_names[1]
        points = {label: (scored[label][nx], scored[label][ny]) for label in runs}
        out_scatter = os.path.join(cfg.out_dir, f"{cfg.tag}_joint_{nx}_vs_{ny}.png")
        plot_joint(points, nx, ny, out_scatter,
                   f"{cfg.operator}: base vs singles vs composed", cfg)
        print(f"[plot] {out_scatter}")
        out_kde = os.path.join(cfg.out_dir, f"{cfg.tag}_kde_{nx}_vs_{ny}.png")
        plot_kde(points, nx, ny, out_kde,
                 f"{cfg.operator}: reward density (base vs singles vs composed)", cfg)
        print(f"[plot] {out_kde}")

    if len(reward_names) >= 3:
        out_grid = os.path.join(cfg.out_dir, f"{cfg.tag}_kdegrid.png")
        plot_kde_grid(scored, reward_names, out_grid,
                      f"{cfg.operator}: pairwise reward density", cfg)
        print(f"[plot] {out_grid}")
        out_min = os.path.join(cfg.out_dir, f"{cfg.tag}_minreward.png")
        min_means = plot_min_reward(scored, reward_names, out_min,
                                    f"{cfg.operator}: worst-objective score (conjunction diagnostic)",
                                    cfg, score_floors=score_floors)
        for label, mv in min_means.items():
            all_summ[label]["min_score_mean"] = mv
        print(f"[plot] {out_min}  (mean worst-objective: "
              + ", ".join(f"{l}={v:.3f}" for l, v in min_means.items()) + ")")
        if len(reward_names) == 3:
            out_tern = os.path.join(cfg.out_dir, f"{cfg.tag}_ternary.png")
            plot_ternary(scored, reward_names, out_tern,
                         f"{cfg.operator}: reward balance (simplex)", cfg,
                         score_floors=score_floors)
            print(f"[plot] {out_tern}")

    if cfg.fcd_enabled:
        fcd_fn = _get_fcd()
        ref = _load_ref_smiles(cfg.fcd_ref_smiles)
        if fcd_fn is not None and ref is not None and len(ref) > 10:
            for label, mols in runs.items():
                smi = mols_to_smiles(mols)
                if len(smi) > 10:
                    all_summ[label]["fcd_vs_ref"] = fcd_fn(smi, ref)
                    print(f"[fcd] {label} vs ref = {all_summ[label]['fcd_vs_ref']:.3f}")
        else:
            print("[fcd] skipped (no backend or no ref).")

    # sample composed WITH model log P(x)
    comp_mols, comp_logp = comp.sample_composed_with_logp(cfg.n_samples)

    # composed target log-reward per molecule = Σ_k w_k log R_k(x)
    # use the component eval rewards named c0.., c1.. (NOT the aggregate MPO),
    # with the composition weights.
    weights = [float(x) for x in cfg.weights.split(",")]
    comp_names = [n for n, _ in comp.eval_rewards]
    # pick the component rewards in the same order as the guides/weights.
    # convention: the eval_rewards after the first (aggregate) are the components.
    # adjust the slice if the eval_rewards ordering differs.
    component_names = comp_names[1:1+len(weights)]
    if len(component_names) == len(weights):
        comp_scored = comp.score(comp_mols)   # dict name -> [N] log-reward
        target_logr = np.zeros(len(comp_mols))
        for w, nm in zip(weights, component_names):
            target_logr += w * np.asarray(comp_scored[nm])
        res = slope_test(comp_logp, target_logr,
                         invalid_logr=cfg.invalid_logr,
                         label="composed",
                         weights_desc="+".join(f"{w:.2f}*{nm}"
                                               for w, nm in zip(weights, component_names)),
                         out_png=os.path.join(cfg.out_dir, f"{cfg.tag}_slope.png"))
        print(f"[slope] slope={res['slope']:.3f}  r={res['r']:.3f}  "
              f"n={res['n']}  -> {res.get('verdict','')}")
        all_summ.setdefault("composed", {})["slope"] = res["slope"]
        all_summ["composed"]["slope_r"] = res["r"]
        with open(os.path.join(cfg.out_dir, f"{cfg.tag}_slope.json"), "w") as f:
            json.dump(res, f, indent=2)

    with open(os.path.join(cfg.out_dir, f"{cfg.tag}_summary.json"), "w") as f:
        json.dump({"config": asdict(cfg), "runs": all_summ}, f, indent=2)
    if cfg.save_mols:
        for label, mols in runs.items():
            torch.save(
                {"mols": mols, "rewards": {n: scored[label][n].tolist() for n in reward_names}},
                os.path.join(cfg.out_dir, f"{cfg.tag}_{label}_mols.pt"))
            smi = mols_to_smiles(mols)
            with open(os.path.join(cfg.out_dir, f"{cfg.tag}_{label}_smiles.txt"), "w") as f:
                f.write("\n".join(smi))
    print(f"[done] outputs in {cfg.out_dir}")

if __name__ == "__main__":
    main()