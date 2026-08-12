"""
Replay buffer for GFlowNet guide training on top of a frozen Quetzal.

Implements the mode-discovery replay strategy from Vemgal, Malkin & Bengio,
"An Empirical Study of the Effectiveness of Using a Replay Buffer on Mode
Discovery in GFlowNets" (arXiv:2307.07674). We store terminal trajectories
(the sampled ATOM sequence + coords + reward) generated on-policy, and mix a
fixed fraction of replayed trajectories into each training batch.

KEY DIFFERENCE from a vanilla RL replay buffer
-----------------------------------------------
The guide policy changes every step, so we CANNOT reuse the log-probs stored at
insertion time -- they are stale. Instead we store only the *atom sequence*
(and coords, needed to re-encode the frozen prior) and RE-ROLL the guide over
that fixed sequence at sample time in a teacher-forced pass. That recomputes
grad-attached logpf under the current guide. This is what makes the replayed
term a valid gradient signal for DB and RTB.

Storage is atom-sequence + coords + scalar reward. Coords are frozen-prior
outputs; storing them avoids re-running the (stochastic, expensive) coordinate
diffusion during replay -- the atoms are what the guide scores, and the coords
only feed encode1/encode2 as fixed context, so reusing the stored coords is
both cheaper and keeps the teacher-forced context identical to when the
trajectory was collected.
"""

import random
import torch


class TrajectoryReplayBuffer:
    """Fixed-capacity buffer of terminal trajectories keyed by reward.

    Each entry: (atoms[L], coords[L,3], stop_step:int, log_reward:float).
    atoms/coords EXCLUDE the leading GEN token (they are already the [:,1:]
    stored form, matching what Molecule(...) receives), so re-rolling must
    re-prepend GEN just as generate_guided does.
    """

    def __init__(self, capacity=10000, strategy="reward", warmup=256):
        assert strategy in ("reward", "uniform")
        self.capacity = capacity
        self.strategy = strategy
        self.warmup = warmup
        self._data = []  # list of dicts

    def __len__(self):
        return len(self._data)

    def ready(self):
        return len(self._data) >= self.warmup

    def add_batch(self, atoms, coords, stop_step, log_reward, valid_mask=None):
        """Insert a batch of terminal trajectories.

        atoms      : LongTensor [B, Lmax]  (already atoms[:,1:] form, GEN-stripped)
        coords     : FloatTensor [B, Lmax, 3]
        stop_step  : LongTensor [B]   index of the STOP-emitting step
        log_reward : FloatTensor [B]
        valid_mask : optional BoolTensor [B]; only insert valid terminals
        """
        atoms = atoms.detach().cpu()
        coords = coords.detach().cpu()
        stop_step = stop_step.detach().cpu()
        log_reward = log_reward.detach().cpu()
        B = atoms.shape[0]
        for i in range(B):
            if valid_mask is not None and not bool(valid_mask[i]):
                continue
            L = int(stop_step[i].item()) + 1
            L = max(1, min(L, atoms.shape[1]))
            self._data.append({
                "atoms": atoms[i, :L].clone(),
                "coords": coords[i, :L].clone(),
                "log_reward": float(log_reward[i].item()),
            })
        self._evict()

    def _evict(self):
        if len(self._data) <= self.capacity:
            return
        if self.strategy == "reward":
            # keep the highest-reward trajectories (mode-seeking retention)
            self._data.sort(key=lambda d: d["log_reward"], reverse=True)
            self._data = self._data[:self.capacity]
        else:
            # uniform: drop random excess (FIFO-ish via random eviction)
            overflow = len(self._data) - self.capacity
            drop = set(random.sample(range(len(self._data)), overflow))
            self._data = [d for i, d in enumerate(self._data) if i not in drop]

    def sample(self, n):
        """Return a list of up to n entries under the configured strategy."""
        if len(self._data) == 0:
            return []
        n = min(n, len(self._data))
        if self.strategy == "reward":
            # reward-prioritized: softmax over log_reward as sampling weights.
            lr = torch.tensor([d["log_reward"] for d in self._data])
            # temperature 1 on log_reward == weight ∝ reward; stabilize by shift
            w = torch.softmax(lr - lr.max(), dim=0)
            idx = torch.multinomial(w, n, replacement=False)
            return [self._data[i] for i in idx.tolist()]
        else:
            return random.sample(self._data, n)


def collate_replayed(entries, device, pad_atom):
    """Pad a list of buffer entries into batched tensors for teacher forcing.

    Returns atoms[B,Lmax], coords[B,Lmax,3], lengths[B], log_reward[B], all on
    `device`. atoms/coords are the GEN-stripped stored form; the teacher-forced
    re-roll re-prepends GEN.
    """
    B = len(entries)
    lengths = [e["atoms"].shape[0] for e in entries]
    Lmax = max(lengths)
    atoms = torch.full((B, Lmax), pad_atom, dtype=torch.long)
    coords = torch.zeros(B, Lmax, 3)
    log_reward = torch.zeros(B)
    for i, e in enumerate(entries):
        L = lengths[i]
        atoms[i, :L] = e["atoms"]
        coords[i, :L] = e["coords"]
        log_reward[i] = e["log_reward"]
    return (atoms.to(device), coords.to(device),
            torch.tensor(lengths, device=device),
            log_reward.to(device))