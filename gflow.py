"""
Guide training on top of a frozen Quetzal prior.

Samples from the tilted distribution p*(x) ~ p_prior(x) * R(x)^beta by training
a small guide network that modifies Quetzal's atom-type logits. The coordinate
diffusion stays frozen throughout, so for a given atom sequence its conditioner
is identical under the guided policy and the prior, its log-density cancels
exactly in the trajectory ratio, and the objective involves only the discrete
atom-type decisions.

Three guide architectures are selectable, all zero-initialised so training
begins at the frozen prior:

    residual   guided = proj_logits(h) + g(h)              (--no_use_hidden_guide)
    temp/gain  guided = proj_logits(h)/T(h) + gamma(h)g(h) (--use_prior_temp
                                                            --use_residual_gain)
    hidden     guided = proj_logits(h + delta(h))          (the default)

and four objectives (--objective): detailed balance, relative trajectory
balance, and the two KL directions.
"""

import os
import glob
import json
import datetime
import argparse
import importlib.util
from dataclasses import dataclass, asdict, fields

import numpy as np
import numpy as np, scipy
if not hasattr(scipy, "histogram"): scipy.histogram = np.histogram
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
torch.set_float32_matmul_precision("medium")
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.rank_zero import rank_zero_only

import wandb

from model import Quetzal
from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from metrics import compute_valid_unique
from reward_fn import build_reward, mol_to_rdkit
from tempgain_guide import TempGainGuide
from replay_buffer import TrajectoryReplayBuffer, collate_replayed
from rdkit import Chem

entity = os.getenv("WANDB_ENTITY")


# ======================= Guided generation =================================

def _generate_guided(self, bsz, guide, sample_temp=1.0, rand_eps=0.0,
                     max_len=None, device="cpu", pbar=False, mask_atoms=None,
                     prefix=None, **kwargs):
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

    max_len = max_len or self.block_size

    if prefix is None:
        atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(bsz, 1, 3, device=device)
    else:
        atoms, coords = prefix
        atoms = atoms.to(device)
        coords = coords.to(device)
        atoms = torch.cat([torch.tensor([GEN], device=device), atoms])
        coords = torch.cat([torch.zeros(1, 3, device=device), coords])
        atoms = atoms.expand(bsz, -1)
        coords = coords.expand(bsz, -1, 3)
        max_len = max_len - atoms.shape[1]

    all_traj = []
    stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
    logp_policy = torch.zeros(bsz, device=device)
    logp_prior = torch.zeros(bsz, device=device)
    uniform = mask.float()
    uniform = uniform / uniform.sum()
    NEG = -1e9

    rng = tqdm.trange(max_len) if pbar else range(max_len)
    for _ in rng:
        idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)

        with torch.no_grad():
            seq = self.encode1(idx, atoms, coords)
            h = seq[:, -1, :]
            prior_logits = self.proj_logits(h)

        if guide is None:
            guided = prior_logits
        elif hasattr(guide, 'guided_logits'):
            try:
                guided = guide.guided_logits(prior_logits, h)   # TempGainGuide (2-arg)
            except TypeError:
                guided = guide.guided_logits(h)                  # HiddenGuide (1-arg, applies proj)
        else:
            guided = prior_logits + guide(h)
        guided = guided.float().masked_fill(~mask, NEG)
        prior_masked = prior_logits.float().masked_fill(~mask, NEG)

        logp_pol = F.log_softmax(guided, dim=-1)
        logp_pri = F.log_softmax(prior_masked, dim=-1)

        with torch.no_grad():
            behav = F.softmax(guided.detach() / sample_temp, dim=-1)
            if rand_eps > 0:
                behav = (1 - rand_eps) * behav + rand_eps * uniform
            next_atom = torch.multinomial(behav, num_samples=1)

        alive = (~stop_mask).float()
        logp_policy = logp_policy + logp_pol.gather(-1, next_atom).squeeze(-1) * alive
        logp_prior = logp_prior + logp_pri.gather(-1, next_atom).squeeze(-1) * alive

        stop_mask = stop_mask | (next_atom.squeeze(-1) == STOP)
        if stop_mask.all():
            break

        atoms = torch.cat([atoms, next_atom], dim=1)

        with torch.no_grad():
            x = self.encode2(atoms[:, 1:], seq)[:, -1, :]
            next_coord, traj = self.sample_coord(x, device=device, **kwargs)
        all_traj.append(traj)
        coords = torch.cat([coords, next_coord.view(bsz, 1, 3)], dim=1)

    if len(all_traj) == 0:
        all_traj = torch.zeros(bsz, 0, 0, 3, device=device)
    else:
        all_traj = torch.stack(all_traj, dim=1)

    mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu")
    info = {"logp_policy": logp_policy, "logp_prior": logp_prior.detach()}
    return mols, all_traj, info


Quetzal.generate_guided = _generate_guided


# ============================ Configuration ================================

@dataclass
class GFNConfig:
    quetzal_ckpt: str = "geom.ckpt"
    train_module: str = "train.py"
    use_ema_prior: bool = True

    name: str = "gfn-geom-peri-comp0--db-beta20"
    devices: int = 1
    num_nodes: int = 1
    # Training seed. Seeds guide initialisation (the hidden layers are random
    # even though the output layer is zero-init), rollout sampling, and the
    # replay buffer, so two runs differing only in `seed` are independent draws
    # of the same configuration. Distinct from final_dump.py's --seed, which
    # only reseeds sampling from an already-trained checkpoint.
    seed: int = 0
    debug: bool = False
    resume_path: str = None
    # Load a bare LogitGuide checkpoint's residual weights into a fresh
    # TempGainGuide's base guide, leaving the temperature and gain heads at
    # identity, so training fine-tunes those on top of an already-trained
    # residual rather than relearning it. Empty means no warm start.
    warm_start_guide: str = ""
    # Which weights to pull from the warm-start checkpoint: "ema"
    # (guide_ema.module.*) or "policy" (guide.*). The EMA weights are the ones
    # that generated the evaluation SMILES.
    warm_start_source: str = "ema"

    dataset: str = "geom"
    bsz: int = 128
    max_len: int = 192
    diff_steps: int = 18
    mask_atoms: str = None

    reward: str = "guacamol_component"
    reward_beta: float = 5.0
    invalid_logr: float = -5.0        
    db_target_clip: float = -6.0      # optional: clamp valid terminal targets >= -8
    reward_target: float = 0.0
    reward_sigma: float = 1.0   # width of the logp/tpsa Gaussian. TPSA needs ~20 (0-140 scale); logp ~1
    reward_smiles: str = None #"hard_osimertinib" | "hard_fexofenadine" | "sitagliptin_replacement" | "perindopril_rings"
    reward_benchmark : str = "perindopril_rings"
    reward_component : int = 0
    force_method: str = "xtb"

    objective: str = "db"          # rtb | vargrad | db
    logz_init: float = 0.0
    ratio_clip: float = 0.0
    # --- DB (detailed balance) flow head ---
    flow_hidden: int = 512          # hidden width of the log-flow head
    flow_layers: int = 2            # depth of the log-flow head
    db_interior_weight: float = 1.0 # weight on interior DB residuals (terminal weight = 1)

    vocab_size: int = 128
    guide_hidden: int = 512
    guide_layers: int = 2
    # Learned per-state temperature on the prior, which softens it, plus a gain
    # that amplifies the residual where the guide is confident. Both are
    # zero-initialised, so this starts at the plain prior-plus-residual.
    use_prior_temp: bool = False
    use_residual_gain: bool = False
    tempgain_hidden: int = 128
    # Inject on the hidden state before proj_logits rather than adding to the
    # output logits. proj_logits amplifies a small hidden delta into a large
    # logit change, because it moves along the directions the projection uses.
    # Mutually exclusive with the temp/gain wrapper, which has its own
    # architecture.
    use_hidden_guide: bool = True
    hidden_guide_out_residual: bool = True

    sample_temp: float = 2.0
    rand_eps: float = 0.2

    steps_per_epoch: int = 100
    max_epochs: int = 6
    lr: float = 1e-4
    logz_lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.99
    wd: float = 0.0
    grad_clip: float = 1.0
    guide_ema_decay: float = 0.999

    # --- Replay buffer (arXiv:2307.07674) -----------------------------------
    # Mixes a fixed fraction of replayed high-reward trajectories into each
    # DB or RTB batch. Replayed trajectories are teacher-forced through the
    # guide so their logpf is gradient-attached under the current policy;
    # log-probs stored at insertion time would be stale. This improves mode
    # discovery without changing the reported bound.
    use_replay: bool = False
    replay_capacity: int = 10000
    replay_strategy: str = "reward"       # "reward" (prioritized) | "uniform"
    replay_fraction: float = 0.25         # share of each batch drawn from buffer
    replay_warmup: int = 256              # min buffer size before replay kicks in
    replay_insert_valid_only: bool = True # only store reward-valid terminals

    # Evaluation
    eval_n: int = 2000
    eval_base: bool = True
    topk_list: str = "1,10,100"

    # Histogram + FCD
    hist_every_n_epochs: int = 5     # 0 disables
    hist_n: int = 0                  # 0 => use eval_n default
    fcd_ref_smiles: str = None       # path to reference SMILES (.txt/.smi)
    fcd_enabled: bool = True

    # Final dump
    final_n: int = 0
    final_dir: str = None

    num_workers: int = 4
    vis_every_n_epochs: int = 5
    save_interval_minutes: int = 10


# ============================ Frozen Quetzal ===============================

def _import_by_path(path, mod_name="quetzal_train"):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_frozen_quetzal(cfg):
    if cfg.quetzal_ckpt is None:
        raise ValueError("Set --quetzal_ckpt to a trained LitQuetzal checkpoint.")
    train_mod = _import_by_path(cfg.train_module)
    lit = train_mod.LitQuetzal.load_from_checkpoint(cfg.quetzal_ckpt, map_location="cpu")
    prior = lit.ema.module if cfg.use_ema_prior else lit.model
    prior.eval()
    for p in prior.parameters():
        p.requires_grad = False
    return prior, lit.config


# ============================ Guide network ================================

class LogitGuide(nn.Module):
    def __init__(self, d_model, vocab_size, hidden, layers):
        super().__init__()
        dims = [d_model] + [hidden] * (layers - 1)
        net = []
        for a, b in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(a, b), nn.SiLU()]
        out = nn.Linear(dims[-1], vocab_size)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        net.append(out)
        self.net = nn.Sequential(*net)

    def forward(self, h):
        return self.net(h)

class LogFlowHead(nn.Module):
    """Scalar per-state flow *residual* f_theta(h) -> R. The full log-flow is
    log F(s) = logp_prior_partial(s) + f_theta(h_s), assembled in the DB loss.
    Zero-init the output so training starts at log F(s) = prior partial (i.e. the
    guide begins as an unbiased continuation of the frozen prior)."""
    def __init__(self, d_model, hidden, layers):
        super().__init__()
        dims = [d_model] + [hidden] * (layers - 1)
        net = []
        for a, b in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(a, b), nn.SiLU()]
        out = nn.Linear(dims[-1], 1)
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        net.append(out)
        self.net = nn.Sequential(*net)
 
    def forward(self, h):
        return self.net(h).squeeze(-1)


# ============================ Helpers ======================================

def _parse_topk(s):
    out = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out


def topk_reward(log_rewards, ks):
    vals = np.asarray(log_rewards, dtype=float)
    vals_sorted = np.sort(vals)[::-1]
    out = {}
    n = len(vals_sorted)
    for k in ks:
        kk = min(k, n)
        out[k] = float(vals_sorted[:kk].mean()) if kk > 0 else float("nan")
    return out


def mols_to_smiles(mols):
    smiles = []
    for m in mols:
        rd = mol_to_rdkit(m)
        if rd is None:
            continue
        try:
            smi = Chem.MolToSmiles(Chem.RemoveHs(rd))
            if smi:
                smiles.append(smi)
        except Exception:
            continue
    return smiles


_FCD_STATE = {"fn": None, "ref": None, "tried": False}


def _get_fcd():
    if _FCD_STATE["tried"]:
        return _FCD_STATE["fn"]
    _FCD_STATE["tried"] = True
    try:
        from fcd_torch import FCD as _FCDT
        scorer = _FCDT(device="cuda" if torch.cuda.is_available() else "cpu")

        def _fn(a, b):
            return float(scorer(a, b))
        _FCD_STATE["fn"] = _fn
        return _fn
    except Exception:
        pass
    try:
        import fcd as _fcd
        model = _fcd.load_ref_model()

        def _fn(a, b):
            return float(_fcd.get_fcd(a, b, model))
        _FCD_STATE["fn"] = _fn
        return _fn
    except Exception:
        pass
    _FCD_STATE["fn"] = None
    return None


def _load_ref_smiles(path):
    if path is None:
        return None
    if _FCD_STATE["ref"] is not None:
        return _FCD_STATE["ref"]
    try:
        with open(path) as f:
            ref = [ln.strip().split()[0] for ln in f if ln.strip()]
        _FCD_STATE["ref"] = ref
        return ref
    except Exception:
        return None


# ============================ Lightning module =============================

class LitGFlowNet(L.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        config = {k: v for k, v in config.items() if k in {f.name for f in fields(GFNConfig)}}
        self.cfg = GFNConfig(**config)
        # guard: some saved checkpoints store diff_steps=None (or a degenerate low
        # value), which crashes sample_coord (torch.arange(None)) or produces
        # degenerate molecules. Coerce to a safe value.
        if getattr(self.cfg, "diff_steps", None) is None or self.cfg.diff_steps < 1:
            print(f"[cfg] diff_steps was {self.cfg.diff_steps!r}; defaulting to 18")
            self.cfg.diff_steps = 18
        self.topks = _parse_topk(self.cfg.topk_list)

        self.frozen, prior_cfg = load_frozen_quetzal(self.cfg)
        d_model = prior_cfg.n_embd

        if self.cfg.use_hidden_guide:
            from hidden_guide import HiddenGuide
            self.guide = HiddenGuide(
                d_model, proj_logits=self.frozen.proj_logits,
                hidden=self.cfg.guide_hidden, layers=self.cfg.guide_layers,
                vocab_size=self.cfg.vocab_size,
                also_output_residual=self.cfg.hidden_guide_out_residual)
            _base_guide = None
            with torch.no_grad():
                _h = torch.randn(4, d_model, device=next(self.guide.parameters()).device)
                assert torch.allclose(self.guide.guided_logits(_h),
                                      self.frozen.proj_logits(_h), atol=1e-4), \
                    'HiddenGuide not identity at init -- check zero-init'
            print("[guide] using HiddenGuide (fix B: hidden-state injection before proj_logits)")
        else:
            _base_guide = LogitGuide(d_model, self.cfg.vocab_size,
                                     self.cfg.guide_hidden, self.cfg.guide_layers)
            self.guide = TempGainGuide(
                _base_guide, d_model, hidden=self.cfg.tempgain_hidden,
                use_temperature=self.cfg.use_prior_temp,
                use_gain=self.cfg.use_residual_gain)
            with torch.no_grad():
                _h = torch.randn(4, d_model)
                _pl = torch.randn(4, self.cfg.vocab_size)
                assert torch.allclose(self.guide.guided_logits(_pl, _h),
                                      _pl + _base_guide(_h), atol=1e-4), \
                    'TempGainGuide not identity at init -- check zero-init'

        # ---- warm start: load a trained LogitGuide residual into the base ----
        if self.cfg.warm_start_guide and _base_guide is not None:
            self._load_warm_start(_base_guide, self.cfg.warm_start_guide,
                                  self.cfg.warm_start_source)

        self.logZ = nn.Parameter(torch.tensor(float(self.cfg.logz_init)))

        # DB needs a per-state log-flow head. Only built when objective == "db".
        if self.cfg.objective == "db":
            self.flow_head = LogFlowHead(d_model, self.cfg.flow_hidden,
                                         self.cfg.flow_layers)
        else:
            self.flow_head = None

        self.reward_fn = build_reward(self.cfg)

        # ---- replay buffer (optional) ----
        self.replay = None
        if self.cfg.use_replay:
            if self.cfg.objective not in ("db", "rtb"):
                print(f"[replay] objective={self.cfg.objective} unsupported; "
                      f"replay disabled (only db/rtb are teacher-forced).")
            else:
                self.replay = TrajectoryReplayBuffer(
                    capacity=self.cfg.replay_capacity,
                    strategy=self.cfg.replay_strategy,
                    warmup=self.cfg.replay_warmup)
                print(f"[replay] enabled: cap={self.cfg.replay_capacity} "
                      f"strategy={self.cfg.replay_strategy} "
                      f"frac={self.cfg.replay_fraction} "
                      f"warmup={self.cfg.replay_warmup} objective={self.cfg.objective}")

        self.guide_ema = AveragedModel(
            self.guide, device="cpu",
            multi_avg_fn=get_ema_multi_avg_fn(self.cfg.guide_ema_decay),
            use_buffers=True,
        )
        self.guide_ema.eval()
        for p in self.guide_ema.parameters():
            p.requires_grad = False

    def _load_warm_start(self, base_guide, ckpt_path, source):
        """Load a LogitGuide's residual weights from `ckpt_path` into
        `base_guide`, the TempGainGuide's base. The temperature and gain heads
        are left at identity, so training fine-tunes those on top.

        A guide checkpoint stores its weights under 'guide.*' (policy) or
        'guide_ema.module.*' (EMA). The prefix is stripped and the rest loaded
        into base_guide, a plain LogitGuide whose parameters are 'net.*'.
        """
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        if source == "ema":
            prefix = "guide_ema.module."
        elif source == "policy":
            prefix = "guide."
        else:
            raise ValueError(f"warm_start_source must be 'ema' or 'policy', got {source!r}")

        # Pull the base-guide weights. Where the checkpoint's guide IS a
        # LogitGuide the keys read 'guide.net.0.weight'; where it is a
        # TempGainGuide the base sits one level deeper at 'guide.guide.net.*'.
        # Try the flat prefix first and fall back to the nested one.
        def collect(pfx):
            return {k[len(pfx):]: v for k, v in sd.items()
                    if k.startswith(pfx) and "n_averaged" not in k
                    and not k[len(pfx):].startswith(("temp.", "gain.", "guide."))}

        sub = collect(prefix)
        # a TempGainGuide checkpoint keeps the residual one level deeper
        nested = prefix + "guide."
        if not sub or all(not kk.startswith("net.") for kk in sub):
            deeper = {k[len(nested):]: v for k, v in sd.items()
                      if k.startswith(nested) and "n_averaged" not in k}
            if deeper:
                sub = deeper

        missing, unexpected = base_guide.load_state_dict(sub, strict=False)
        real_missing = [m for m in missing if "n_averaged" not in m]
        with torch.no_grad():
            wnorm = sum(p.float().norm().item() for p in base_guide.parameters())
        print(f"[warm_start] loaded base residual from {ckpt_path} "
              f"(source={source}): base weight-norm={wnorm:.3f}, "
              f"missing={len(real_missing)}, unexpected={len(unexpected)}")
        if wnorm < 1e-6:
            print("[warm_start] WARNING: base weight-norm ~0 -- nothing loaded. "
                  "Check the ckpt prefix/source; residual is effectively untrained.")
        if real_missing:
            print(f"[warm_start] missing keys (first few): {real_missing[:6]}")

    def compute_log_reward(self, mols) -> torch.Tensor:
        vals = [self.reward_fn(m) for m in mols]
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    def rollout(self, n, guide="policy", sample_temp=None, rand_eps=None,
                with_reward=True):
        self.frozen.eval()
        if guide == "policy":
            guide_mod = self.guide
        elif guide == "ema":
            guide_mod = self.guide_ema.module
        else:
            guide_mod = guide

        mols, _, info = self.frozen.generate_guided(
            n, guide=guide_mod,
            sample_temp=self.cfg.sample_temp if sample_temp is None else sample_temp,
            rand_eps=self.cfg.rand_eps if rand_eps is None else rand_eps,
            max_len=self.cfg.max_len, device=self.device, pbar=self.cfg.debug,
            mask_atoms=self.cfg.mask_atoms, num_steps=self.cfg.diff_steps,
        )
        mols = mols.unbatch()
        out = {"mols": mols,
               "logp_policy": info["logp_policy"],
               "logp_prior": info["logp_prior"]}
        if with_reward:
            out["log_reward"] = self.compute_log_reward(mols)
        return out

    def rollout_chunked(self, n, guide, chunk=500, with_reward=True):
        all_mols, all_lr = [], []
        done = 0
        while done < n:
            b = min(chunk, n - done)
            out = self.rollout(b, guide=guide, sample_temp=1.0, rand_eps=0.0,
                               with_reward=with_reward)
            all_mols.extend(out["mols"])
            if with_reward:
                all_lr.append(out["log_reward"])
            done += b
        res = {"mols": all_mols}
        if with_reward:
            res["log_reward"] = torch.cat(all_lr) if all_lr else torch.zeros(0)
        return res

    
    def _db_rollout_states(self, prior, guide, bsz, max_len, device, mask_atoms=None,
                           sample_temp=1.0, rand_eps=0.0, **coord_kwargs):
        """Roll out bsz trajectories, RETAINING gradient on the guide policy.

        Only the frozen prior (encode/logits/coord) and the multinomial draw run
        under no_grad. The guide's forward log-probs stay attached so DB trains
        BOTH the policy (P_F) and the flow head jointly.
        """
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
        stop_step = torch.full((bsz,), -1, dtype=torch.long, device=device)

        hs_list, logpf_chosen_list, logpf_stop_list = [], [], []
        logprior_step_list, step_mask_list = [], []

        for t in range(max_len):
            idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)

            # ---- frozen prior: no grad ----
            with torch.no_grad():
                seq = prior.encode1(idx, atoms, coords)
                h = seq[:, -1, :]                       # [B, d] state hidden
                prior_logits = prior.proj_logits(h)

            # ---- guide policy: gradient attached ----
            if guide is None:
                guided = prior_logits
            elif hasattr(guide, 'guided_logits'):
                try:
                    guided = guide.guided_logits(prior_logits, h)   # TempGainGuide
                except TypeError:
                    guided = guide.guided_logits(h)                  # HiddenGuide
            else:
                guided = prior_logits + guide(h)
            guided = guided.float().masked_fill(~mask, NEG)
            prior_masked = prior_logits.float().masked_fill(~mask, NEG)
            logp_pol = F.log_softmax(guided, dim=-1)     # attached to guide
            logp_pri = F.log_softmax(prior_masked, dim=-1)  # frozen -> effectively const

            # ---- sampling: no grad ----
            with torch.no_grad():
                behav = F.softmax(guided.detach() / sample_temp, dim=-1)
                if rand_eps > 0:
                    behav = (1 - rand_eps) * behav + rand_eps * uniform
                next_atom = torch.multinomial(behav, num_samples=1)  # [B,1]

            alive = (~stop_mask)
            hs_list.append(h)                                   # detached is fine for flow head
            logpf_chosen_list.append(logp_pol.gather(-1, next_atom).squeeze(-1))  # GRAD
            logpf_stop_list.append(logp_pol[:, STOP])                              # GRAD
            logprior_step_list.append(logp_pri.gather(-1, next_atom).squeeze(-1).detach())
            step_mask_list.append(alive.clone())

            emitted_stop = (next_atom.squeeze(-1) == STOP) & alive
            stop_step = torch.where(emitted_stop, torch.full_like(stop_step, t), stop_step)
            stop_mask = stop_mask | (next_atom.squeeze(-1) == STOP)
            if stop_mask.all():
                break

            atoms = torch.cat([atoms, next_atom], dim=1)
            with torch.no_grad():
                x = prior.encode2(atoms[:, 1:], seq)[:, -1, :]
                next_coord, _ = prior.sample_coord(x, device=device, **coord_kwargs)
            coords = torch.cat([coords, next_coord.view(bsz, 1, 3)], dim=1)

        T = len(hs_list)
        hs = torch.stack(hs_list, dim=1)
        logpf_chosen = torch.stack(logpf_chosen_list, dim=1)
        logpf_stop = torch.stack(logpf_stop_list, dim=1)
        logprior_step = torch.stack(logprior_step_list, dim=1)
        step_mask = torch.stack(step_mask_list, dim=1)
        stop_step = torch.where(stop_step < 0, torch.full_like(stop_step, T - 1), stop_step)

        from chem import Molecule
        mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu")
        return {
            "mols": mols,
            "hs": hs, "logpf_chosen": logpf_chosen, "logpf_stop": logpf_stop,
            "logprior_step": logprior_step, "step_mask": step_mask, "stop_step": stop_step,
            # stored (GEN-stripped) tensors for the replay buffer
            "atoms_stored": atoms[:, 1:].detach(),
            "coords_stored": coords[:, 1:].detach(),
        }

    def _teacher_force_states(self, prior, guide, atoms_in, coords_in, lengths,
                              mask_atoms=None):
        """Re-roll the guide over STORED atom sequences (no sampling).

        atoms_in  : [B, Lmax] GEN-stripped stored atoms (buffer form)
        coords_in : [B, Lmax, 3] stored coords (frozen-prior context)
        lengths   : [B] true length of each sequence (== stop_step + 1)

        Returns the same dict shape as `_db_rollout_states` so DB and RTB losses
        can both consume it. logpf_chosen/logpf_stop are grad-attached under the
        CURRENT guide; the frozen prior and stored coords are fixed context, so
        no coordinate diffusion runs during replay.
        """
        device = atoms_in.device
        B, Lmax = atoms_in.shape
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
        NEG = -1e9

        # re-prepend GEN + zero coord, exactly as generate_guided does
        gen_col = torch.full((B, 1), GEN, dtype=torch.long, device=device)
        atoms = torch.cat([gen_col, atoms_in], dim=1)             # [B, Lmax+1]
        coords = torch.cat([torch.zeros(B, 1, 3, device=device),
                            coords_in], dim=1)                     # [B, Lmax+1, 3]

        hs_list, logpf_chosen_list, logpf_stop_list = [], [], []
        logprior_step_list, step_mask_list = [], []
        # stop_step in the re-rolled frame == lengths - 1 (0-indexed over emitted atoms)
        stop_step = (lengths - 1).clamp(min=0)

        for t in range(Lmax):
            # context = GEN + first t emitted atoms; predict emitted atom at t
            idx = torch.arange(t + 1, device=device).expand(B, -1)
            with torch.no_grad():
                seq = prior.encode1(idx, atoms[:, :t + 1], coords[:, :t + 1])
                h = seq[:, -1, :]
                prior_logits = prior.proj_logits(h)

            if guide is None:
                guided = prior_logits
            elif hasattr(guide, 'guided_logits'):
                try:
                    guided = guide.guided_logits(prior_logits, h)   # TempGainGuide
                except TypeError:
                    guided = guide.guided_logits(h)                  # HiddenGuide
            else:
                guided = prior_logits + guide(h)
            guided = guided.float().masked_fill(~mask, NEG)
            prior_masked = prior_logits.float().masked_fill(~mask, NEG)
            logp_pol = F.log_softmax(guided, dim=-1)
            logp_pri = F.log_softmax(prior_masked, dim=-1)

            target = atoms[:, t + 1]                     # the stored emitted atom
            alive = (t < lengths)                        # steps within this seq
            hs_list.append(h)
            logpf_chosen_list.append(logp_pol.gather(-1, target.unsqueeze(-1)).squeeze(-1))
            logpf_stop_list.append(logp_pol[:, STOP])
            logprior_step_list.append(
                logp_pri.gather(-1, target.unsqueeze(-1)).squeeze(-1).detach())
            step_mask_list.append(alive.clone())

        T = len(hs_list)
        hs = torch.stack(hs_list, dim=1)
        logpf_chosen = torch.stack(logpf_chosen_list, dim=1)
        logpf_stop = torch.stack(logpf_stop_list, dim=1)
        logprior_step = torch.stack(logprior_step_list, dim=1)
        step_mask = torch.stack(step_mask_list, dim=1)
        stop_step = stop_step.clamp(max=T - 1)

        from chem import Molecule
        mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu")
        return {
            "mols": mols,
            "hs": hs, "logpf_chosen": logpf_chosen, "logpf_stop": logpf_stop,
            "logprior_step": logprior_step, "step_mask": step_mask,
            "stop_step": stop_step,
        }

    def _replay_insert_from_mols(self, mols, log_reward):
        """Store unbatched Molecules into the replay buffer (RTB path, where
        rollout() discards per-step state). Reconstructs padded atom/coord tensors
        from the Molecule list."""
        valid_mask = (log_reward > self.cfg.invalid_logr + 0.1) \
            if self.cfg.replay_insert_valid_only else None
        atom_list = [torch.as_tensor(m.atoms).long().flatten() for m in mols]
        coord_list = [torch.as_tensor(m.coords).float().reshape(-1, 3) for m in mols]
        Lmax = max((a.shape[0] for a in atom_list), default=1)
        B = len(atom_list)
        atoms = torch.full((B, Lmax), PAD, dtype=torch.long)
        coords = torch.zeros(B, Lmax, 3)
        stop_step = torch.zeros(B, dtype=torch.long)
        for i, (a, c) in enumerate(zip(atom_list, coord_list)):
            L = a.shape[0]
            atoms[i, :L] = a
            cc = min(L, c.shape[0])
            coords[i, :cc] = c[:cc]
            stop_step[i] = max(L - 1, 0)
        self.replay.add_batch(atoms, coords, stop_step,
                              log_reward.detach().cpu(), valid_mask=valid_mask)

    def _replay_db_loss(self):
        """Sample the buffer, teacher-force under the current guide, and return a
        DB loss on the replayed trajectories (or None if not ready)."""
        if not (self.replay.ready() and self.cfg.replay_fraction > 0):
            return None
        n_replay = int(self.cfg.bsz * self.cfg.replay_fraction)
        entries = self.replay.sample(n_replay)
        if not entries:
            return None
        r_atoms, r_coords, r_len, r_logr = collate_replayed(
            entries, self.device, pad_atom=PAD)
        r_roll = self._teacher_force_states(
            self.frozen, self.guide, r_atoms, r_coords, r_len,
            mask_atoms=self.cfg.mask_atoms)
        r_loss, _ = self.db_loss_from_rollout(
            self.flow_head, r_roll, r_logr, self.cfg.reward_beta,
            invalid_logr=self.cfg.invalid_logr,
            target_clip=getattr(self.cfg, "db_target_clip", None))
        return r_loss

    def _replay_rtb_loss(self):
        """Sample the buffer, teacher-force, and return an RTB loss on the
        replayed trajectories (or None if not ready)."""
        if not (self.replay.ready() and self.cfg.replay_fraction > 0):
            return None
        n_replay = int(self.cfg.bsz * self.cfg.replay_fraction)
        entries = self.replay.sample(n_replay)
        if not entries:
            return None
        r_atoms, r_coords, r_len, r_logr = collate_replayed(
            entries, self.device, pad_atom=PAD)
        r_roll = self._teacher_force_states(
            self.frozen, self.guide, r_atoms, r_coords, r_len,
            mask_atoms=self.cfg.mask_atoms)
        m = r_roll["step_mask"].float()
        # trajectory ratio sum_t (logpf_chosen - logprior_step) over alive steps
        r_ratio = ((r_roll["logpf_chosen"] - r_roll["logprior_step"]) * m).sum(dim=1)
        if self.cfg.ratio_clip > 0:
            r_ratio = r_ratio.clamp(-self.cfg.ratio_clip, self.cfg.ratio_clip)
        r_logR = self.cfg.reward_beta * r_logr
        r_delta = r_ratio - r_logR
        return (self.logZ + r_delta).pow(2).mean()

    def db_loss_from_rollout(self, flow_head, roll, log_reward, beta,
                             invalid_logr=None, target_clip=None):
        """Detailed-Balance loss (tilted target, P_B = 1) with INVALID TERMINALS
        MASKED OUT of the terminal condition.

        invalid_logr : the reward floor; terminals whose log_reward is at/near it
                       are treated as genuine invalids and excluded from the
                       terminal loss (they carry no reward to reconstruct).
        target_clip  : optional lower bound on the per-sample terminal target
                       beta*log_reward, so a rare very-negative VALID reward can't
                       dominate. None = no clip.
        """
        hs = roll["hs"]
        logpf_chosen = roll["logpf_chosen"]
        logpf_stop = roll["logpf_stop"]
        logprior_step = roll["logprior_step"]
        step_mask = roll["step_mask"]
        stop_step = roll["stop_step"]
        B, T, d = hs.shape
        device = hs.device

        f = flow_head(hs)
        logprior_partial = torch.cumsum(logprior_step, dim=1) - logprior_step
        logF = logprior_partial + f
        logF_next = torch.cat([logF[:, 1:], logF[:, -1:].detach()], dim=1)

        # ---- interior residuals: valid for t < stop_step ----
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        interior_valid = (t_idx < stop_step.unsqueeze(1)) & step_mask
        interior_res = (logF + logpf_chosen - logF_next) * interior_valid.float()
        n_int = interior_valid.float().sum().clamp(min=1.0)
        interior_loss = (interior_res.pow(2).sum()) / n_int

        # ---- terminal residual: at state x = stop_step, VALID terminals only ----
        x = stop_step.clamp(min=0, max=T - 1)
        bidx = torch.arange(B, device=device)
        logF_x = logF[bidx, x]
        logpf_stop_x = logpf_stop[bidx, x]
        logprior_partial_x = logprior_partial[bidx, x]

        logR = log_reward
        if target_clip is not None:
            logR = logR.clamp(min=target_clip)   # e.g. -8 so target >= -8*beta
        logR_b = beta * logR

        # mask: keep only terminals whose reward is a genuine VALID score.
        if invalid_logr is None:
            valid_term = torch.ones_like(log_reward, dtype=torch.bool)
        else:
            valid_term = log_reward > (invalid_logr + 0.1)

        terminal_res_full = logF_x + logpf_stop_x - (logprior_partial_x + logR_b)
        if valid_term.any():
            terminal_res = terminal_res_full[valid_term]
            terminal_loss = terminal_res.pow(2).mean()
        else:
            # no valid molecules this batch -> no terminal signal; keep graph alive
            terminal_loss = (terminal_res_full * 0.0).sum()
            terminal_res = terminal_res_full

        total = interior_loss + terminal_loss
        stats = {
            "db/interior_loss": interior_loss.detach(),
            "db/terminal_loss": terminal_loss.detach() if torch.is_tensor(terminal_loss)
                                else torch.tensor(float(terminal_loss)),
            "db/terminal_res_abs_mean": terminal_res.detach().abs().mean(),
            "db/mean_logF_x": logF_x.detach().mean(),
            "db/frac_valid_terminal": valid_term.float().mean().detach(),
        }
        return total, stats

    def training_step(self, batch, batch_idx):
        # ---------------- Detailed Balance branch ----------------
        if self.cfg.objective == "db":
            self.frozen.eval()
            roll = self._db_rollout_states(
                self.frozen, self.guide, self.cfg.bsz, self.cfg.max_len,
                self.device, mask_atoms=self.cfg.mask_atoms,
                sample_temp=self.cfg.sample_temp, rand_eps=self.cfg.rand_eps,
                num_steps=self.cfg.diff_steps,
            )
            mols = roll["mols"].unbatch()
            log_reward = self.compute_log_reward(mols)      # [B]
            loss, db_stats = self.db_loss_from_rollout(
                self.flow_head, roll, log_reward, self.cfg.reward_beta,
                invalid_logr=self.cfg.invalid_logr,
                target_clip=getattr(self.cfg, "db_target_clip", None))
            # weight interior vs terminal if requested (default 1:1)
            if self.cfg.db_interior_weight != 1.0:
                loss = (self.cfg.db_interior_weight * db_stats["db/interior_loss"]
                        + db_stats["db/terminal_loss"])

            # ---- replay buffer: insert on-policy terminals, mix replayed loss ----
            if self.replay is not None:
                valid_mask = (log_reward > self.cfg.invalid_logr + 0.1) \
                    if self.cfg.replay_insert_valid_only else None
                self.replay.add_batch(
                    roll["atoms_stored"], roll["coords_stored"],
                    roll["stop_step"], log_reward, valid_mask=valid_mask)
                r_loss = self._replay_db_loss()
                if r_loss is not None:
                    frac = self.cfg.replay_fraction
                    loss = (1 - frac) * loss + frac * r_loss
                    self.log("train/replay_loss", r_loss.detach(), prog_bar=False)
                self.log("train/replay_size", float(len(self.replay)), prog_bar=False)

            self.guide_ema.update_parameters(self.guide)

            chem_valid_frac = float(np.mean(
                [mol_to_rdkit(m) is not None for m in mols]))
            reward_valid_frac = float(np.mean(
                [lr > self.cfg.invalid_logr + 0.1 for lr in log_reward.tolist()]))
            self.log_dict({
                "train/loss": loss,
                "train/log_reward_mean": log_reward.mean(),
                "train/log_reward_max": log_reward.max(),
                "train/valid_frac": chem_valid_frac,
                "train/reward_valid_frac": reward_valid_frac,
                **db_stats,
            }, prog_bar=True)
            return loss

        # ---------------- KL baselines (same guide, different loss) --------------
        # Separates the loss from the guidance mechanism: an identical
        # prior + guide(h) residual, trained instead by a direct KL to the
        # tilted target p*(x) ~ p_prior(x) * R(x)^beta at the trajectory level,
        # which is comparable to DB's terminal target. If these reach the same
        # flip rates, the bound does not depend on the choice of objective.
        if self.cfg.objective in ("revkl", "fwdkl"):
            out = self.rollout(self.cfg.bsz, guide="policy")
            lp_guide = out["logp_policy"]           # log q(x)   (attached, grad)
            lp_prior = out["logp_prior"]            # log p_prior(x) (detached)
            beta = self.cfg.reward_beta
            logR = out["log_reward"]                # log R(x)
            # tilted (unnormalized) target log-prob: log p*(x) = log p_prior + beta logR
            log_ptar = lp_prior + beta * logR       # up to the (unknown) constant -log Z

            if self.cfg.objective == "revkl":
                # Reverse KL  KL(q || p*) = E_q[ log q - log p* ].
                # On-policy REINFORCE-style: minimize E_q[ (log q - log p*) ].
                # Use the score-function estimator with a baseline (mean) to reduce
                # variance; stop-grad the reward term (it's the target, not the model).
                adv = (lp_guide - log_ptar).detach()
                adv = adv - adv.mean()              # baseline
                loss = (lp_guide * adv).mean()
                # also log the raw KL value for monitoring
                kl_val = (lp_guide - log_ptar).mean().detach()
            else:  # fwdkl
                # Forward KL  KL(p* || q) = E_{p*}[ -log q ] + const.
                # We don't have samples from p*; use self-normalized importance
                # sampling with PRIOR proposals: weight prior samples by R^beta and
                # maximize weighted log q. Since `out` is sampled from the GUIDE, we
                # importance-correct with w ∝ p*(x)/q(x) = exp(log_ptar - log q).
                logw = (log_ptar - lp_guide).detach()
                logw = logw - torch.logsumexp(logw, dim=0)   # self-normalize
                w = logw.exp()
                loss = -(w * lp_guide).sum()        # weighted NLL under the guide
                kl_val = -(w * lp_guide).sum().detach()

            self.guide_ema.update_parameters(self.guide)
            mols = out["mols"]
            chem_valid_frac = float(np.mean([mol_to_rdkit(m) is not None for m in mols]))
            self.log_dict({
                "train/loss": loss,
                "train/kl_value": kl_val,
                "train/log_reward_mean": logR.mean(),
                "train/log_reward_max": logR.max(),
                "train/valid_frac": chem_valid_frac,
                "train/mean_logq_minus_logprior": (lp_guide - lp_prior).mean().detach(),
            }, prog_bar=True)
            return loss

        # ---------------- RTB / VarGrad branch ----------------
        out = self.rollout(self.cfg.bsz, guide="policy")
        ratio = out["logp_policy"] - out["logp_prior"]
        if self.cfg.ratio_clip > 0:
            ratio = ratio.clamp(-self.cfg.ratio_clip, self.cfg.ratio_clip)
        logR = self.cfg.reward_beta * out["log_reward"]
        delta = ratio - logR
        if self.cfg.objective == "rtb":
            loss = (self.logZ + delta).pow(2).mean()
        elif self.cfg.objective == "vargrad":
            loss = (delta - delta.mean().detach()).pow(2).mean()
        else:
            raise ValueError(f"Unknown objective {self.cfg.objective!r}")

        # ---- replay buffer (RTB only; vargrad has no logZ target) ----
        if self.replay is not None and self.cfg.objective == "rtb":
            self._replay_insert_from_mols(out["mols"], out["log_reward"])
            r_loss = self._replay_rtb_loss()
            if r_loss is not None:
                frac = self.cfg.replay_fraction
                loss = (1 - frac) * loss + frac * r_loss
                self.log("train/replay_loss", r_loss.detach(), prog_bar=False)
            self.log("train/replay_size", float(len(self.replay)), prog_bar=False)

        self.guide_ema.update_parameters(self.guide)
        # chemical validity: fraction of molecules that convert to a sanitized RDKit
        # mol (this is what compute_valid_unique measures), INDEPENDENT of the reward.
        # A run whose reward is misconfigured (e.g. bad sigma -> every reward = floor)
        # will still show high chem_valid_frac here, so the two are disentangled.
        chem_valid_frac = float(np.mean(
            [mol_to_rdkit(m) is not None for m in out["mols"]]))
        # reward-scored fraction: molecules that scored above the invalid floor. Low
        # values here with high chem_valid_frac == the REWARD is failing, not the model.
        reward_valid_frac = float(np.mean(
            [lr > self.cfg.invalid_logr + 0.1 for lr in out["log_reward"].tolist()]))
        self.log_dict({
            "train/loss": loss,
            "train/logZ": self.logZ.detach(),
            "train/log_reward_mean": out["log_reward"].mean(),
            "train/log_reward_max": out["log_reward"].max(),
            "train/log_ratio_mean": ratio.mean(),
            "train/valid_frac": chem_valid_frac,
            "train/reward_valid_frac": reward_valid_frac,
        }, prog_bar=True)
        return loss

    def _eval_block(self, tag, guide):
        cfg = self.cfg
        n_eval = cfg.eval_n if cfg.eval_n > 0 else (2000 if cfg.dataset == "qm9" else 500)
        out = self.rollout(n_eval, guide=guide, sample_temp=1.0, rand_eps=0.0)
        valid, unique = compute_valid_unique(out["mols"])
        logr = out["log_reward"]
        metrics = {
            f"{tag}/validity": valid,
            f"{tag}/uniqueness": unique,
            f"{tag}/valid_unique": valid * unique,
            f"{tag}/log_reward_mean": float(logr.mean()),
        }
        for k, v in topk_reward(logr.tolist(), self.topks).items():
            metrics[f"{tag}/log_reward_top{k}"] = v
        return metrics, valid, logr

    @torch.no_grad()
    def on_train_epoch_end(self):
        cfg = self.cfg
        if self.global_rank != 0:
            return

        if self.current_epoch % cfg.vis_every_n_epochs == 0:
            log = {}
            g_metrics, _, _ = self._eval_block("eval", guide="ema")
            log.update(g_metrics)
            if cfg.eval_base:
                b_metrics, _, _ = self._eval_block("eval_base", guide=None)
                log.update(b_metrics)
                log["eval_delta/log_reward_mean"] = (
                    g_metrics["eval/log_reward_mean"] - b_metrics["eval_base/log_reward_mean"])
                log["eval_delta/validity"] = (
                    g_metrics["eval/validity"] - b_metrics["eval_base/validity"])
                for k in self.topks:
                    log[f"eval_delta/log_reward_top{k}"] = (
                        g_metrics[f"eval/log_reward_top{k}"]
                        - b_metrics[f"eval_base/log_reward_top{k}"])
            self.log_dict(log, rank_zero_only=True)

        if cfg.hist_every_n_epochs > 0 and (self.current_epoch % cfg.hist_every_n_epochs == 0):
            self._hist_and_fcd()

    @torch.no_grad()
    def _hist_and_fcd(self):
        cfg = self.cfg
        n = cfg.hist_n if cfg.hist_n > 0 else (cfg.eval_n if cfg.eval_n > 0
                                               else (2000 if cfg.dataset == "qm9" else 500))
        guided = self.rollout(n, guide="ema", sample_temp=1.0, rand_eps=0.0)
        base = self.rollout(n, guide=None, sample_temp=1.0, rand_eps=0.0)
        g_logr = guided["log_reward"].cpu().numpy()
        b_logr = base["log_reward"].cpu().numpy()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            lo = float(min(g_logr.min(), b_logr.min()))
            hi = float(max(g_logr.max(), b_logr.max()))
            bins = np.linspace(lo, hi, 50)
            ax.hist(b_logr, bins=bins, alpha=0.5, label="base", density=True)
            ax.hist(g_logr, bins=bins, alpha=0.5, label="guided", density=True)
            ax.set_xlabel("log reward"); ax.set_ylabel("density")
            ax.set_title(f"reward dist (epoch {self.current_epoch})")
            ax.legend()
            wandb.log({"hist/log_reward": wandb.Image(fig),
                       "trainer/global_step": self.global_step})
            plt.close(fig)
        except Exception as e:
            print(f"[hist] failed: {e}")

        if cfg.fcd_enabled:
            try:
                fcd_fn = _get_fcd()
                if fcd_fn is None:
                    print("[fcd] no FCD backend (pip install fcd_torch). Skipping.")
                else:
                    g_smiles = mols_to_smiles(guided["mols"])
                    b_smiles = mols_to_smiles(base["mols"])
                    log = {}
                    if len(g_smiles) > 10 and len(b_smiles) > 10:
                        log["fcd/guided_vs_base"] = fcd_fn(g_smiles, b_smiles)
                    ref = _load_ref_smiles(cfg.fcd_ref_smiles)
                    if ref is not None and len(ref) > 10:
                        if len(g_smiles) > 10:
                            log["fcd/guided_vs_ref"] = fcd_fn(g_smiles, ref)
                        if len(b_smiles) > 10:
                            log["fcd/base_vs_ref"] = fcd_fn(b_smiles, ref)
                    if log:
                        self.log_dict(log, rank_zero_only=True)
            except Exception as e:
                print(f"[fcd] failed: {e}")

    @rank_zero_only
    def _dump_final(self):
        cfg = self.cfg
        out_dir = cfg.final_dir or f"logs/quetzal-gfn/{cfg.name}/final"
        os.makedirs(out_dir, exist_ok=True)
        print(f"[final] generating {cfg.final_n} molecules each for base + guided ...")

        for tag, guide in (("guided", "ema"), ("base", None)):
            res = self.rollout_chunked(cfg.final_n, guide=guide, chunk=500, with_reward=True)
            mols = res["mols"]
            logr = res["log_reward"].cpu().numpy()
            smiles = mols_to_smiles(mols)

            torch.save({"mols": mols, "log_reward": logr},
                       os.path.join(out_dir, f"{tag}_molecules.pt"))
            with open(os.path.join(out_dir, f"{tag}_smiles.txt"), "w") as f:
                f.write("\n".join(smiles))
            with open(os.path.join(out_dir, f"{tag}_rewards.json"), "w") as f:
                json.dump(logr.tolist(), f)

            valid, unique = compute_valid_unique(mols)
            tk = topk_reward(logr.tolist(), self.topks)
            summary = {
                "n": len(mols), "n_valid_smiles": len(smiles),
                "validity": valid, "uniqueness": unique,
                "log_reward_mean": float(logr.mean()),
                **{f"log_reward_top{k}": v for k, v in tk.items()},
            }
            with open(os.path.join(out_dir, f"{tag}_summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[final] {tag}: {summary}")

        print(f"[final] saved to {out_dir}")

    def on_fit_end(self):
        if self.cfg.final_n > 0:
            self._dump_final()

    def configure_optimizers(self):
        groups = [{"params": self.guide.parameters(), "lr": self.cfg.lr,
                   "weight_decay": self.cfg.wd}]
        if self.cfg.objective == "rtb":
            groups.append({"params": [self.logZ], "lr": self.cfg.logz_lr,
                           "weight_decay": 0.0})
        if self.cfg.objective == "db":
            # flow head learns at the same lr as the guide; no logZ under DB
            groups.append({"params": self.flow_head.parameters(),
                           "lr": self.cfg.lr, "weight_decay": self.cfg.wd})
        return torch.optim.AdamW(groups, betas=(self.cfg.beta1, self.cfg.beta2), eps=1e-12)


# ============================ Data + main ==================================

class StepDataModule(L.LightningDataModule):
    def __init__(self, steps_per_epoch, num_workers):
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers

    def train_dataloader(self):
        ds = TensorDataset(torch.zeros(self.steps_per_epoch, 1))
        return DataLoader(ds, batch_size=1, num_workers=self.num_workers, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser()
    for field in GFNConfig.__dataclass_fields__.values():
        if isinstance(field.default, bool):
            if field.default is False:
                parser.add_argument(f"--{field.name}", dest=field.name, action="store_true")
            else:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action="store_false")
            parser.set_defaults(**{field.name: field.default})
        else:
            argtype = type(field.default) if field.default is not None else str
            parser.add_argument(f"--{field.name}", type=argtype, default=field.default)
    return GFNConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    # before the module is built: the guide's hidden layers are randomly
    # initialised, so seeding after construction would leave them uncontrolled
    L.seed_everything(cfg.seed, workers=True)
    lit = LitGFlowNet(asdict(cfg))

    @rank_zero_only
    def print_once():
        print(cfg)
        guide_params = sum(p.numel() for p in lit.guide.parameters())
        print(f"Guide parameters: {guide_params/1e6:.2f}M | objective={cfg.objective} "
              f"| top-k={lit.topks} | hist_every={cfg.hist_every_n_epochs} "
              f"| final_n={cfg.final_n}")

    print_once()

    checkpoint_dir = f"logs/quetzal-gfn/{cfg.name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    wandb_logger = WandbLogger(save_dir="logs", project="quetzal-gfn", entity=entity,
                               name=cfg.name, config=asdict(cfg), offline=False)

    def latest_ckpt(d):
        ckpts = glob.glob(os.path.join(d, "*.ckpt"))
        return max(ckpts, key=os.path.getmtime) if ckpts else None

    resume_path = cfg.resume_path or latest_ckpt(checkpoint_dir)

    print(f"Resuming from: {resume_path}")

    ckpt_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        train_time_interval=datetime.timedelta(minutes=cfg.save_interval_minutes),
        save_last="link", save_on_train_epoch_end=True)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.devices, num_nodes=cfg.num_nodes,
        logger=wandb_logger, log_every_n_steps=10,
        gradient_clip_val=cfg.grad_clip if cfg.grad_clip > 0 else None,
        precision="bf16-mixed", enable_progress_bar=cfg.debug,
        callbacks=[ckpt_cb], num_sanity_val_steps=0)

    dm = StepDataModule(cfg.steps_per_epoch, cfg.num_workers)
    trainer.fit(lit, datamodule=dm, ckpt_path=resume_path)
    wandb_logger.finalize(status="success")