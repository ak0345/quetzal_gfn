"""
RTB (Relative Trajectory Balance) fine-tuning of Quetzal ITSELF, as a capacity
control for the saturated-prior-ceiling result.

WHY THIS EXISTS
---------------
`gflow.py` trains a small *guide* that adds a residual to frozen atom-type
logits. The thesis is that this can't steer hard MPO rewards because the
residual (norm ~5-80) is dwarfed by the prior logits (norm ~6841). That is a
CAPACITY claim. This script removes the capacity limit by training the model's
own weights under the same RTB objective, which cleanly separates:

  * fine-tuned model steers osim  -> the ceiling was real and architectural
  * fine-tuned model also fails   -> capacity was never binding; the sparse
                                     reward / unfittable target is the cause

FINE-TUNING SCOPES  (--finetune_scope)
--------------------------------------
  proj  : ONLY `proj_logits` (optionally LoRA-wrapped).
          encode1 / encode2 / simple_mlp stay frozen, so z_prefix for a given
          atom sequence is IDENTICAL between policy and prior. Therefore
          p(coords | atoms) cancels exactly in the RTB ratio and the atom-only
          log-ratio is EXACT. This is the minimal unbounded-capacity
          intervention on precisely the object the ceiling argument is about.
          >>> This is the scientifically clean comparison to the guides. <<<

  atom  : proj_logits + the encode1 trunk (blocks1, embeddings, wpe).
          Now z_prefix drifts, so p(coords|atoms) is NO LONGER identical and
          the atom-only ratio is an APPROXIMATION. We log
          `diag/zprefix_drift` (mean L2 between policy and frozen z_prefix) so
          the size of the violation is measurable rather than assumed.

  full  : everything, including blocks2 and the coordinate diffusion MLP.
          Same approximation as `atom`, larger drift. Use for "can the model
          do it at all", not for a clean RTB claim.

An exact treatment of the coordinate term would need the ODE log-density
(`Quetzal.log_density`, 120 steps of jacrev) inside the training loop, which is
not affordable per batch. `proj` sidesteps the issue entirely -- prefer it for
headline numbers and treat atom/full as capacity probes.

LoRA  (--lora_rank > 0)
-----------------------
Wraps matched nn.Linear modules with a zero-init low-rank adapter (identity at
step 0, like the guide's zero-init residual). Sweeping rank in {4,16,64,full}
turns "does it steer" from a binary into a curve of steering vs. trainable
parameter norm -- a much stronger figure than "guided ~= base".
`--lora_targets` is a comma-separated list of substrings matched against module
names; the matched list is printed at startup so you can verify it against your
actual attention.py naming.

MOLECULE RECORDING  (--record_dir)
----------------------------------
Every molecule generated during training is appended, in generation order, to
`molecules.jsonl` with its global index `i` (= oracle call number), epoch, step,
canonical SMILES and log-reward. 3D structures go to `shard_*.pt`. This makes
`harvest_eval.py` possible WITHOUT retraining: take the first N records, dedupe,
rescore, report top-k. See harvest_eval.py.

Note on budget arithmetic: bsz * steps_per_epoch molecules per epoch. At the
defaults (128 * 100) that is 12,800 -- so a 10,000-call budget is reached
partway through epoch 1. Train longer than the budget you intend to report.

EXAMPLES
--------
# clean capacity control: proj_logits only, exact RTB, osimertinib MPO
python rtb_finetune.py --name ft-proj-osim-b10 \
  --quetzal_ckpt geom.ckpt --finetune_scope proj \
  --reward guacamol --reward_smiles hard_osimertinib \
  --objective rtb --reward_beta 10 --beta_start 1 --beta_anneal_epochs 4 \
  --max_epochs 20 --record_dir records/ft-proj-osim-b10

# LoRA capacity dial on the same axis
for R in 4 16 64; do
  python rtb_finetune.py --name ft-lora$R-osim-b10 --finetune_scope proj \
    --lora_rank $R --lora_targets proj_logits \
    --reward guacamol --reward_smiles hard_osimertinib --reward_beta 10 \
    --record_dir records/ft-lora$R-osim-b10
done

# full fine-tune (approximate ratio -- watch diag/zprefix_drift)
python rtb_finetune.py --name ft-full-osim-b10 --finetune_scope full \
  --logp_grad_frac 0.25 --bsz 64 \
  --reward guacamol --reward_smiles hard_osimertinib --reward_beta 10 \
  --record_dir records/ft-full-osim-b10

# dense-reward sanity check (should move if the loop works at all)
python rtb_finetune.py --name ft-proj-nitrogen-b10 --finetune_scope proj \
  --reward nitrogen_count --reward_beta 10 --record_dir records/ft-proj-n-b10
"""

import os
import re
import glob
import json
import math
import copy
import datetime
import argparse
import importlib.util
from dataclasses import dataclass, asdict, fields

import numpy as np
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

from rdkit import Chem

from model import Quetzal
from chem import Molecule, GEN, STOP, PAD, QM9_MASK
from metrics import compute_valid_unique
from reward_fn import build_reward, mol_to_rdkit

entity = os.getenv("WANDB_ENTITY")


# ============================== Configuration ==============================

@dataclass
class FTConfig:
    # ---- prior / checkpoint ----
    quetzal_ckpt: str = "geom.ckpt"
    train_module: str = "train.py"
    use_ema_prior: bool = True          # start policy from the EMA weights too

    name: str = "ft-proj-osim-b10"
    devices: int = 1
    num_nodes: int = 1
    debug: bool = False
    resume_path: str = None

    # ---- sampling ----
    dataset: str = "geom"
    bsz: int = 64
    max_len: int = 192
    diff_steps: int = 18
    mask_atoms: str = None
    sample_temp: float = 1.0            # behaviour policy temperature
    rand_eps: float = 0.0               # eps-uniform exploration mixin

    # ---- reward (mirrors gflow.py / reward_fn.py) ----
    reward: str = "guacamol"
    reward_beta: float = 10.0
    beta_start: float = 1.0             # anneal beta_start -> reward_beta
    beta_anneal_epochs: int = 0         # 0 = no anneal (start at reward_beta)
    invalid_logr: float = -5.0
    reward_target: float = 0.0
    reward_sigma: float = 1.0
    reward_smiles: str = "hard_osimertinib"
    reward_benchmark: str = "perindopril_rings"
    reward_component: int = 0
    reward_formula: str = None
    force_method: str = "xtb"

    # ---- objective ----
    objective: str = "rtb"              # rtb | vargrad
    logz_init: float = 0.0
    ratio_clip: float = 0.0

    # ---- what gets trained ----
    finetune_scope: str = "proj"        # proj | atom | full
    lora_rank: int = 0                  # 0 = full-rank (direct) fine-tuning
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_targets: str = "proj_logits"   # comma-separated name substrings

    # ---- optimisation ----
    lr: float = 1e-5                    # small: these are pretrained weights
    logz_lr: float = 1e-2               # logZ needs a much larger lr
    trunk_lr_mult: float = 0.1          # trunk lr = lr * this (atom/full scope)
    beta1: float = 0.9
    beta2: float = 0.99
    wd: float = 0.0
    grad_clip: float = 1.0
    warmup_steps: int = 100
    logp_grad_frac: float = 1.0         # unbiased grad subsampling over tokens
    use_ema_policy: bool = False        # EMA over the policy (memory-hungry)
    ema_decay: float = 0.999

    steps_per_epoch: int = 100
    max_epochs: int = 20

    # ---- molecule recording (for harvest_eval.py) ----
    record_dir: str = None             # None disables
    record_smiles: bool = True
    record_coords: bool = True
    record_shard_size: int = 5000

    # ---- eval ----
    eval_n: int = 0                     # per-epoch policy-vs-prior eval (0 = off)
    eval_base: bool = True
    topk_list: str = "1,10,100"
    final_n: int = 0

    # ---- misc ----
    num_workers: int = 4
    save_interval_minutes: int = 15
    save_trainable_only: bool = True    # keep checkpoints small


# ============================ LoRA ==========================================

class LoRALinear(nn.Module):
    """Zero-init low-rank adapter. Identity at step 0, so training starts
    exactly at the prior (same cold-start property as the zero-init guide)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)                      # -> delta W = 0 at init
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.base(x)
        lora = (self.drop(x) @ self.A.t()) @ self.B.t()
        return out + lora * self.scaling

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features


def apply_lora(model: nn.Module, patterns, r: int, alpha: float, dropout: float):
    """Replace every nn.Linear whose qualified name contains any pattern."""
    if r <= 0:
        return []
    matched = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(p in name for p in patterns):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(module, r, alpha, dropout))
        matched.append(name)
    return matched


# ============================ Model loading =================================

def _import_by_path(path, mod_name="quetzal_train"):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_prior_and_policy(cfg):
    """Two copies of Quetzal from the same checkpoint: a frozen reference for
    the RTB ratio, and a trainable policy."""
    if cfg.quetzal_ckpt is None:
        raise ValueError("Set --quetzal_ckpt to a trained LitQuetzal checkpoint.")
    train_mod = _import_by_path(cfg.train_module)
    lit = train_mod.LitQuetzal.load_from_checkpoint(cfg.quetzal_ckpt, map_location="cpu")

    src = lit.ema.module if cfg.use_ema_prior else lit.model
    frozen = src
    policy = copy.deepcopy(src)

    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad = False
    return frozen, policy, lit.config


# --- which parameter groups belong to which scope -------------------------

ENCODE1_PREFIXES = ("embed_atoms", "embed_coords", "embed_fourier",
                    "embed_scalars", "wpe", "blocks1")
ENCODE2_PREFIXES = ("blocks2",)
DIFFUSION_PREFIXES = ("simple_mlp",)
HEAD_PREFIXES = ("proj_logits",)


def set_trainable(policy: nn.Module, scope: str):
    """Freeze everything, then unfreeze by scope. Returns (n_trainable, names)."""
    for p in policy.parameters():
        p.requires_grad = False

    if scope == "proj":
        allow = HEAD_PREFIXES
    elif scope == "atom":
        allow = HEAD_PREFIXES + ENCODE1_PREFIXES
    elif scope == "full":
        allow = HEAD_PREFIXES + ENCODE1_PREFIXES + ENCODE2_PREFIXES + DIFFUSION_PREFIXES
    else:
        raise ValueError(f"finetune_scope must be proj|atom|full, got {scope!r}")

    names = []
    for name, p in policy.named_parameters():
        # LoRA adapters live inside the module they wrap, so prefix matching
        # covers them; `.base.` params stay frozen via LoRALinear.__init__.
        if name.startswith(allow) and ".base." not in name:
            p.requires_grad = True
            names.append(name)
    n = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return n, names


def trunk_param_names(scope):
    """Params that get lr * trunk_lr_mult (everything that isn't the head)."""
    if scope == "proj":
        return ()
    return ENCODE1_PREFIXES + ENCODE2_PREFIXES + DIFFUSION_PREFIXES


# ============================ Helpers =======================================

def _parse_topk(s):
    return [int(t) for t in str(s).split(",") if t.strip()]


def topk_reward(log_rewards, ks):
    lr = sorted(log_rewards, reverse=True)
    return {k: float(np.mean(lr[:k])) if lr else float("nan") for k in ks}


def mol_to_smiles(m):
    rd = mol_to_rdkit(m)
    if rd is None:
        return None
    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(rd))
        return smi or None
    except Exception:
        return None


def build_atom_mask(mask_atoms, device):
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
    return mask


# ============================ Molecule recorder =============================

class MoleculeRecorder:
    """Append-only log of every molecule generated during training, in order.

    molecules.jsonl : one row per molecule
        {"i": global oracle index, "epoch": e, "step": s,
         "smiles": str|null, "log_reward": float, "n_atoms": int}
    shard_XXXXX.pt  : {"idx": [...], "atoms": [...], "coords": [...]}
    meta.json       : config echo + running count

    `i` IS the oracle call index -- harvest_eval.py slices on it to enforce a
    budget. Order is preserved on resume by counting existing rows.
    """

    def __init__(self, out_dir, shard_size=5000, save_coords=True, meta=None):
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.dir, "molecules.jsonl")
        self.shard_size = shard_size
        self.save_coords = save_coords
        self.count = 0
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path) as f:
                self.count = sum(1 for _ in f)
            print(f"[record] resuming; {self.count} molecules already logged")
        self.fh = open(self.jsonl_path, "a", buffering=1)
        self._buf_idx, self._buf_atoms, self._buf_coords = [], [], []
        self._shard_id = len(glob.glob(os.path.join(self.dir, "shard_*.pt")))
        if meta is not None:
            with open(os.path.join(self.dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)

    def add_batch(self, mols, log_rewards, epoch, step, smiles=None):
        lrs = log_rewards.detach().cpu().tolist() if torch.is_tensor(log_rewards) \
            else list(log_rewards)
        for j, m in enumerate(mols):
            smi = smiles[j] if smiles is not None else None
            a = m.atoms.detach().cpu() if torch.is_tensor(m.atoms) else torch.as_tensor(m.atoms)
            n_atoms = int((a > 0).sum().item())
            self.fh.write(json.dumps({
                "i": self.count, "epoch": int(epoch), "step": int(step),
                "smiles": smi, "log_reward": float(lrs[j]), "n_atoms": n_atoms,
            }) + "\n")
            if self.save_coords:
                self._buf_idx.append(self.count)
                self._buf_atoms.append(a.to(torch.int16))
                c = m.coords.detach().cpu() if torch.is_tensor(m.coords) \
                    else torch.as_tensor(m.coords)
                self._buf_coords.append(c.to(torch.float16))
            self.count += 1
        if len(self._buf_idx) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self._buf_idx:
            return
        path = os.path.join(self.dir, f"shard_{self._shard_id:05d}.pt")
        torch.save({"idx": self._buf_idx, "atoms": self._buf_atoms,
                    "coords": self._buf_coords}, path)
        self._shard_id += 1
        self._buf_idx, self._buf_atoms, self._buf_coords = [], [], []

    def close(self):
        self.flush()
        try:
            self.fh.close()
        except Exception:
            pass


# ============================ Lightning module ==============================

class LitRTBFineTune(L.LightningModule):

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        config = {k: v for k, v in config.items() if k in {f.name for f in fields(FTConfig)}}
        self.cfg = FTConfig(**config)
        if getattr(self.cfg, "diff_steps", None) is None or self.cfg.diff_steps < 1:
            self.cfg.diff_steps = 18
        self.topks = _parse_topk(self.cfg.topk_list)

        self.frozen, self.policy, prior_cfg = load_prior_and_policy(self.cfg)
        self.d_model = prior_cfg.n_embd

        # LoRA first, so set_trainable's prefix match also catches adapters
        patterns = [p.strip() for p in self.cfg.lora_targets.split(",") if p.strip()]
        self.lora_names = apply_lora(self.policy, patterns, self.cfg.lora_rank,
                                     self.cfg.lora_alpha, self.cfg.lora_dropout)
        if self.cfg.lora_rank > 0:
            if not self.lora_names:
                raise ValueError(
                    f"--lora_rank={self.cfg.lora_rank} but no nn.Linear matched "
                    f"{patterns!r}. Print `[n for n,m in model.named_modules()]` "
                    f"and pick real substrings (attention layer names vary).")
            print(f"[lora] rank={self.cfg.lora_rank} alpha={self.cfg.lora_alpha} "
                  f"wrapped {len(self.lora_names)} Linear(s):")
            for n in self.lora_names[:12]:
                print(f"       {n}")
            if len(self.lora_names) > 12:
                print(f"       ... (+{len(self.lora_names)-12} more)")

        n_train, train_names = set_trainable(self.policy, self.cfg.finetune_scope)

        # a LoRA adapter outside the scope's allow-list stays frozen and does
        # nothing -- silently halving your intended capacity dial
        orphan = [n for n in self.lora_names
                  if not any(t.startswith(n.split(".")[0]) or n.startswith(t)
                             for t in train_names)]
        if orphan:
            print(f"[lora] WARNING: {len(orphan)} adapter(s) sit outside "
                  f"finetune_scope={self.cfg.finetune_scope} and will NOT "
                  f"train: {orphan[:6]}. Widen the scope or the targets.")
        n_total = sum(p.numel() for p in self.policy.parameters())
        print(f"[scope] {self.cfg.finetune_scope}: {n_train/1e6:.3f}M trainable "
              f"/ {n_total/1e6:.1f}M total ({100*n_train/max(n_total,1):.2f}%)")
        if self.cfg.finetune_scope != "proj":
            print("[scope] WARNING: encode1 is trainable, so p(coords|atoms) "
                  "drifts from the prior and the atom-only RTB ratio is an "
                  "APPROXIMATION. Watch diag/zprefix_drift.")
        if not train_names:
            raise ValueError("nothing is trainable -- check finetune_scope/lora_targets")

        self.logZ = nn.Parameter(torch.tensor(float(self.cfg.logz_init)))
        self.reward_fn = build_reward(self.cfg)

        self.policy_ema = None
        if self.cfg.use_ema_policy:
            self.policy_ema = AveragedModel(
                self.policy, device="cpu",
                multi_avg_fn=get_ema_multi_avg_fn(self.cfg.ema_decay),
                use_buffers=True)
            self.policy_ema.eval()
            for p in self.policy_ema.parameters():
                p.requires_grad = False

        self.recorder = None
        if self.cfg.record_dir:
            self.recorder = MoleculeRecorder(
                self.cfg.record_dir, shard_size=self.cfg.record_shard_size,
                save_coords=self.cfg.record_coords, meta=asdict(self.cfg))

        self._share_trunk = (self.cfg.finetune_scope == "proj")

    # -------------------- beta annealing --------------------

    @property
    def beta(self):
        if self.cfg.beta_anneal_epochs <= 0:
            return self.cfg.reward_beta
        frac = min(1.0, self.current_epoch / float(self.cfg.beta_anneal_epochs))
        return self.cfg.beta_start + frac * (self.cfg.reward_beta - self.cfg.beta_start)

    # -------------------- reward --------------------

    def compute_log_reward(self, mols):
        vals = [self.reward_fn(m) for m in mols]
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    # -------------------- sampling --------------------

    def sample(self, bsz, policy=None, sample_temp=None, rand_eps=None,
               grad=True, pbar=False):
        """Autoregressive rollout from `policy`, accumulating BOTH the policy
        log-prob (grad-attached when grad=True) and the frozen-prior log-prob.

        Only the ATOM-TYPE decisions enter the log-ratio. Under
        finetune_scope=proj this is exact: the coordinate conditioner z_prefix
        is produced by frozen modules from a frozen `seq`, so p(coords|atoms)
        is identical under policy and prior and cancels. Under atom/full it is
        an approximation (see module docstring).
        """
        policy = policy if policy is not None else self.policy
        device = self.device
        st = self.cfg.sample_temp if sample_temp is None else sample_temp
        eps = self.cfg.rand_eps if rand_eps is None else rand_eps

        mask = build_atom_mask(self.cfg.mask_atoms, device)
        uniform = mask.float()
        uniform = uniform / uniform.sum()
        NEG = -1e9

        atoms = torch.full((bsz, 1), GEN, dtype=torch.long, device=device)
        coords = torch.zeros(bsz, 1, 3, device=device)

        logp_policy = torch.zeros(bsz, device=device)
        logp_prior = torch.zeros(bsz, device=device)
        stop_mask = torch.zeros(bsz, dtype=torch.bool, device=device)
        drift_sum, drift_n = 0.0, 0

        rho = float(self.cfg.logp_grad_frac)
        rng = tqdm.trange(self.cfg.max_len) if pbar else range(self.cfg.max_len)

        for _ in rng:
            idx = torch.arange(atoms.shape[1], device=device).expand(bsz, -1)

            if self._share_trunk:
                # encode1 is byte-identical between policy and frozen -> one pass
                with torch.no_grad():
                    seq = self.frozen.encode1(idx, atoms, coords)
                    h = seq[:, -1, :]
                    prior_logits = self.frozen.proj_logits(h)
                if grad:
                    pol_logits = policy.proj_logits(h)
                else:
                    with torch.no_grad():
                        pol_logits = policy.proj_logits(h)
                seq_for_coords = seq
            else:
                ctx = torch.enable_grad() if grad else torch.no_grad()
                with ctx:
                    seq = policy.encode1(idx, atoms, coords)
                    h = seq[:, -1, :]
                    pol_logits = policy.proj_logits(h)
                with torch.no_grad():
                    fseq = self.frozen.encode1(idx, atoms, coords)
                    prior_logits = self.frozen.proj_logits(fseq[:, -1, :])
                    drift_sum += (seq[:, -1, :].detach() - fseq[:, -1, :]).norm(dim=-1).mean().item()
                    drift_n += 1
                seq_for_coords = seq

            pol_masked = pol_logits.float().masked_fill(~mask, NEG)
            pri_masked = prior_logits.float().masked_fill(~mask, NEG)
            logp_pol = F.log_softmax(pol_masked, dim=-1)
            logp_pri = F.log_softmax(pri_masked, dim=-1)

            with torch.no_grad():
                behav = F.softmax(pol_masked.detach() / st, dim=-1)
                if eps > 0:
                    behav = (1 - eps) * behav + eps * uniform
                next_atom = torch.multinomial(behav, num_samples=1)

            alive = (~stop_mask).float()
            step_pol = logp_pol.gather(-1, next_atom).squeeze(-1) * alive
            step_pri = logp_pri.gather(-1, next_atom).squeeze(-1) * alive

            if grad and rho < 1.0:
                # Unbiased gradient subsampling over timesteps: the VALUE of the
                # sum stays exact, but grad flows through a random rho-fraction
                # of steps, rescaled by 1/rho. E[grad] equals the full gradient,
                # and peak memory drops ~rho x for long sequences.
                keep = (torch.rand(bsz, device=device) < rho).float()
                step_pol = step_pol.detach() + (step_pol - step_pol.detach()) * (keep / rho)

            logp_policy = logp_policy + step_pol
            logp_prior = logp_prior + step_pri

            stop_mask = stop_mask | (next_atom.squeeze(-1) == STOP)
            if stop_mask.all():
                break

            atoms = torch.cat([atoms, next_atom], dim=1)

            with torch.no_grad():
                z = policy.encode2(atoms[:, 1:], seq_for_coords)[:, -1, :]
                next_coord, _ = policy.sample_coord(
                    z, device=device, num_steps=self.cfg.diff_steps)
            coords = torch.cat([coords, next_coord.view(bsz, 1, 3)], dim=1)

        mols = Molecule(atoms=atoms[:, 1:], coords=coords[:, 1:]).to("cpu")
        info = {
            "logp_policy": logp_policy,
            "logp_prior": logp_prior.detach(),
            "zprefix_drift": (drift_sum / drift_n) if drift_n else 0.0,
        }
        return mols, info

    @torch.no_grad()
    def sample_chunked(self, n, policy=None, chunk=250):
        all_mols, all_lr = [], []
        done = 0
        while done < n:
            b = min(chunk, n - done)
            mols, _ = self.sample(b, policy=policy, sample_temp=1.0,
                                  rand_eps=0.0, grad=False)
            mols = mols.unbatch()
            all_mols.extend(mols)
            all_lr.append(self.compute_log_reward(mols))
            done += b
        return all_mols, torch.cat(all_lr) if all_lr else torch.zeros(0)

    # -------------------- training --------------------

    def training_step(self, batch, batch_idx):
        self.frozen.eval()
        mols_b, info = self.sample(self.cfg.bsz, grad=True)
        mols = mols_b.unbatch()
        log_reward = self.compute_log_reward(mols)

        ratio = info["logp_policy"] - info["logp_prior"]
        if self.cfg.ratio_clip > 0:
            ratio = ratio.clamp(-self.cfg.ratio_clip, self.cfg.ratio_clip)

        beta = self.beta
        logR = beta * log_reward
        delta = ratio - logR

        if self.cfg.objective == "rtb":
            loss = (self.logZ + delta).pow(2).mean()
        elif self.cfg.objective == "vargrad":
            loss = (delta - delta.mean().detach()).pow(2).mean()
        else:
            raise ValueError(f"Unknown objective {self.cfg.objective!r}")

        # ---- record every generated molecule, in order ----
        smiles = None
        if self.recorder is not None:
            if self.cfg.record_smiles:
                smiles = [mol_to_smiles(m) for m in mols]
            self.recorder.add_batch(mols, log_reward, self.current_epoch,
                                    batch_idx, smiles=smiles)

        if self.policy_ema is not None:
            self.policy_ema.update_parameters(self.policy)

        if smiles is not None:
            chem_valid = float(np.mean([s is not None for s in smiles]))
        else:
            chem_valid = float(np.mean([mol_to_rdkit(m) is not None for m in mols]))
        reward_valid = float(np.mean(
            [lr > self.cfg.invalid_logr + 0.1 for lr in log_reward.tolist()]))

        self.log_dict({
            "train/loss": loss,
            "train/logZ": self.logZ.detach(),
            "train/beta": torch.tensor(float(beta)),
            "train/log_reward_mean": log_reward.mean(),
            "train/log_reward_max": log_reward.max(),
            "train/log_ratio_mean": ratio.mean().detach(),
            "train/valid_frac": chem_valid,
            "train/reward_valid_frac": reward_valid,
            "diag/zprefix_drift": torch.tensor(float(info["zprefix_drift"])),
            "diag/n_recorded": float(self.recorder.count) if self.recorder else 0.0,
        }, prog_bar=True)
        return loss

    # -------------------- eval --------------------

    def _eval_block(self, tag, policy):
        mols, lr = self.sample_chunked(self.cfg.eval_n, policy=policy)
        valid, unique = compute_valid_unique(mols)
        tk = topk_reward(lr.tolist(), self.topks)
        out = {f"{tag}/log_reward_mean": float(lr.mean()),
               f"{tag}/validity": valid, f"{tag}/uniqueness": unique}
        out.update({f"{tag}/log_reward_top{k}": v for k, v in tk.items()})
        return out

    def on_train_epoch_end(self):
        if self.recorder is not None:
            self.recorder.flush()
        if self.cfg.eval_n <= 0 or self.global_rank != 0:
            return
        logs = self._eval_block("eval_policy", self.policy)
        if self.cfg.eval_base:
            logs.update(self._eval_block("eval_base", self.frozen))
            for k in ("log_reward_mean", *[f"log_reward_top{k}" for k in self.topks]):
                a, b = logs.get(f"eval_policy/{k}"), logs.get(f"eval_base/{k}")
                if a is not None and b is not None:
                    logs[f"delta/{k}"] = a - b
        self.log_dict(logs, rank_zero_only=True)

    def on_fit_end(self):
        if self.recorder is not None:
            self.recorder.close()
        if self.cfg.final_n > 0 and self.global_rank == 0:
            out_dir = f"logs/quetzal-ft/{self.cfg.name}/final"
            os.makedirs(out_dir, exist_ok=True)
            for tag, pol in (("policy", self.policy), ("base", self.frozen)):
                mols, lr = self.sample_chunked(self.cfg.final_n, policy=pol)
                smis = [s for s in (mol_to_smiles(m) for m in mols) if s]
                with open(os.path.join(out_dir, f"{tag}_smiles.txt"), "w") as f:
                    f.write("\n".join(smis))
                json.dump({"n": len(mols), "log_reward_mean": float(lr.mean()),
                           **{f"top{k}": v for k, v in
                              topk_reward(lr.tolist(), self.topks).items()}},
                          open(os.path.join(out_dir, f"{tag}_summary.json"), "w"),
                          indent=2)
            print(f"[final] wrote {out_dir}")

    # -------------------- optim / ckpt --------------------

    def configure_optimizers(self):
        head, trunk = [], []
        tp = trunk_param_names(self.cfg.finetune_scope)
        for name, p in self.policy.named_parameters():
            if not p.requires_grad:
                continue
            (trunk if (tp and name.startswith(tp)) else head).append(p)

        groups = [{"params": head, "lr": self.cfg.lr, "weight_decay": self.cfg.wd}]
        if trunk:
            groups.append({"params": trunk,
                           "lr": self.cfg.lr * self.cfg.trunk_lr_mult,
                           "weight_decay": self.cfg.wd})
        if self.cfg.objective == "rtb":
            # logZ must move much faster than the weights or it lags the reward
            # scale and the residual looks flat for the wrong reason.
            groups.append({"params": [self.logZ], "lr": self.cfg.logz_lr,
                           "weight_decay": 0.0})

        opt = torch.optim.AdamW(groups, betas=(self.cfg.beta1, self.cfg.beta2), eps=1e-12)
        if self.cfg.warmup_steps > 0:
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min(1.0, (s + 1) / self.cfg.warmup_steps))
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        return opt

    def on_save_checkpoint(self, ck):
        if not self.cfg.save_trainable_only:
            return
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        ck["state_dict"] = {k: v for k, v in ck["state_dict"].items() if k in keep}

    def on_load_checkpoint(self, ck):
        # trainable-only checkpoints: backfill frozen weights from the already
        # constructed modules so Lightning's strict load succeeds
        sd = ck.get("state_dict", {})
        cur = self.state_dict()
        for k, v in cur.items():
            if k not in sd:
                sd[k] = v
        ck["state_dict"] = sd


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
    for field in FTConfig.__dataclass_fields__.values():
        if isinstance(field.default, bool):
            if field.default is False:
                parser.add_argument(f"--{field.name}", dest=field.name, action="store_true")
            else:
                parser.add_argument(f"--no_{field.name}", dest=field.name, action="store_false")
            parser.set_defaults(**{field.name: field.default})
        else:
            argtype = type(field.default) if field.default is not None else str
            parser.add_argument(f"--{field.name}", type=argtype, default=field.default)
    return FTConfig(**vars(parser.parse_args()))


if __name__ == "__main__":
    cfg = parse_args()
    lit = LitRTBFineTune(asdict(cfg))

    @rank_zero_only
    def print_once():
        print(cfg)
        n_tr = sum(p.numel() for p in lit.policy.parameters() if p.requires_grad)
        per_epoch = cfg.bsz * cfg.steps_per_epoch
        print(f"Trainable: {n_tr/1e6:.3f}M | scope={cfg.finetune_scope} "
              f"| lora_rank={cfg.lora_rank} | objective={cfg.objective} "
              f"| beta {cfg.beta_start}->{cfg.reward_beta} over {cfg.beta_anneal_epochs} ep")
        print(f"Oracle calls: {per_epoch}/epoch, {per_epoch*cfg.max_epochs} total "
              f"(a 10k budget lands at epoch {10000/max(per_epoch,1):.2f})")

    print_once()

    ckpt_dir = f"logs/quetzal-ft/{cfg.name}/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    wandb_logger = WandbLogger(save_dir="logs", project="quetzal-ft", entity=entity,
                               name=cfg.name, config=asdict(cfg), offline=False)

    def latest_ckpt(d):
        ck = glob.glob(os.path.join(d, "*.ckpt"))
        return max(ck, key=os.path.getmtime) if ck else None

    resume_path = cfg.resume_path or latest_ckpt(ckpt_dir)
    print(f"Resuming from: {resume_path}")

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
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