"""
hang_guard.py -- keep a long run from silently freezing, and make it obvious
where it froze if it does.

THE FAILURE THIS ADDRESSES: training stops progressing while the process stays
alive and holds its GPU memory. No exception, no collapse, no OOM. That is
almost always a CPU-side call that never returns -- for this codebase, most
likely RDKit bond perception from 3D coordinates (rdDetermineBonds searches over
bond orders and charges and can blow up combinatorially on a large or dense
structure), RDKit ring perception on a fused cage, or an xtb subprocess with no
timeout. The GPU sits idle because the training loop is blocked in reward
evaluation.

THREE LAYERS, use all of them:

  1. install_faulthandler()  -- `kill -USR1 <pid>` on a hung process prints the
     stack of every thread. Diagnosis without attaching a debugger.
  2. call_with_timeout()     -- a hard per-molecule ceiling on reward
     evaluation. A molecule that takes longer than the ceiling is scored as
     invalid rather than being allowed to stop the run.
  3. StallGuard              -- a watchdog that notices no batch has completed
     in N minutes, dumps the stacks, flushes what it can, and exits with a
     distinctive code so a supervisor can restart from the last checkpoint.

Layer 2 is the fix; layers 1 and 3 are so that if something ELSE hangs you find
out in minutes rather than discovering a dead run the next morning.

WHAT THE GUARD COSTS YOU. A timed-out molecule is scored at the invalid floor,
so it joins the same bucket as one that fails bond perception -- which is what
it is. But that means a guard firing often silently reshapes the reward
distribution, so the timeout count is logged as `train/reward_timeouts` and
printed periodically. If that number is not ~0, the results are affected and the
timeout needs raising, not ignoring. The same applies with more force to
`max_atoms` (see guarded_reward).
"""
import os
import sys
import time
import signal
import faulthandler
import threading

STALL_EXIT_CODE = 17          # distinctive: a supervisor can retry on this
TIME_LIMIT_EXIT_CODE = 18     # stopped at --max_train_hours, resumable

# Both codes mean "not finished, but the checkpoint is sound, so re-invoking
# resumes rather than restarts". A driver should therefore NOT record such a run
# as complete. Any other non-zero code is a real failure.
RESUMABLE_EXIT_CODES = (STALL_EXIT_CODE, TIME_LIMIT_EXIT_CODE)


def max_time_spec(hours):
    """Lightning's Trainer(max_time=...) wants "DD:HH:MM:SS". Returns None for a
    non-positive value, which disables the limit."""
    if not hours or float(hours) <= 0:
        return None
    total = int(round(float(hours) * 3600))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d:02d}:{h:02d}:{m:02d}:{sec:02d}"


def finish_or_exit(trainer, max_epochs, label=""):
    """Exit 18 when training stopped short of max_epochs because of the wall-clock
    limit, so the driver leaves the run resumable instead of marking it done.

    Lightning stops at a batch boundary and runs the checkpoint callback first, so
    the checkpoint on disk is consistent and `ckpt_path=` picks it up next time.
    """
    done = getattr(trainer, "current_epoch", None)
    if done is None or done >= max_epochs:
        return
    print(f"[guard] {label or 'run'} stopped at epoch {done}/{max_epochs} on the "
          f"wall-clock limit; checkpoint saved, re-run to resume", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    raise SystemExit(TIME_LIMIT_EXIT_CODE)

# faulthandler keeps only the file descriptor, so the Python file object must be
# kept alive or the dump file ends up empty
_KEEP_ALIVE = []


# ------------------------------------------------------------------ layer 1

def install_faulthandler(log_dir=None):
    """Enable fault handling and a SIGUSR1 stack dump.

    After this, `kill -USR1 <pid>` on a frozen run prints every thread's stack
    to stderr (and to a file if log_dir is given) WITHOUT killing the process,
    so you can identify the hang and then decide what to do about it.
    """
    faulthandler.enable()
    _fh = None
    path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"faulthandler-{os.getpid()}.log")
        # keep a module-level reference: faulthandler holds the fd, not the
        # Python object, so a garbage-collected handle silently produces an
        # empty dump file
        _fh = open(path, "a", buffering=1)
        _KEEP_ALIVE.append(_fh)
        faulthandler.enable(file=_fh)

    def _dump(signum, frame):
        # dump to BOTH stderr and the file: stderr is what you see in the
        # terminal, the file is what survives a lost ssh session
        faulthandler.dump_traceback(all_threads=True)
        if _fh is not None:
            _fh.write(f"\n===== SIGUSR1 dump {time.strftime('%Y-%m-%d %H:%M:%S')} "
                      f"=====\n")
            _fh.flush()
            faulthandler.dump_traceback(file=_fh, all_threads=True)
            _fh.flush()

    if not hasattr(signal, "SIGUSR1"):
        print("[guard] SIGUSR1 unavailable on this platform; stack dumps disabled")
        return
    if threading.current_thread() is not threading.main_thread():
        # signal.signal is main-thread only; installing the handler is optional,
        # so degrade rather than take the process down
        print("[guard] not on the main thread; SIGUSR1 handler not installed")
        return
    signal.signal(signal.SIGUSR1, _dump)
    where = f" -> {path}" if path else ""
    print(f"[guard] stack dumps on SIGUSR1{where}  (kill -USR1 {os.getpid()})")


# ------------------------------------------------------------------ layer 2

class _Timeout(Exception):
    pass


def _can_use_sigalrm():
    """SIGALRM is delivered to the main thread only, and signal.signal raises
    from anywhere else. Callers fall back to running unguarded rather than
    turning a hang guard into a crash."""
    return (hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread())


_warned_no_alarm = [False]


def call_with_timeout(fn, *args, seconds=20, default=None, label="", **kwargs):
    """Run fn with a hard wall-clock ceiling; return `default` if it overruns.

    Uses SIGALRM, so it only interrupts code that returns to the Python
    interpreter periodically. That covers RDKit's Python-level loops and
    subprocess waits. A pure C extension that never yields cannot be
    interrupted this way -- for those, see run_in_subprocess below.

    Off the main thread SIGALRM is unavailable, so fn runs unguarded and a
    warning is printed once. In this codebase reward evaluation happens in
    training_step, which Lightning runs on the main thread, so the guard is
    live where it matters.

    Nesting is not supported: an inner call resets the outer call's timer.
    """
    if seconds is None or seconds <= 0:
        return fn(*args, **kwargs)

    if not _can_use_sigalrm():
        if not _warned_no_alarm[0]:
            _warned_no_alarm[0] = True
            print(f"[guard] SIGALRM unavailable off the main thread; "
                  f"{label or 'call'} runs unguarded", flush=True)
        return fn(*args, **kwargs)

    def _handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        return fn(*args, **kwargs)
    except _Timeout:
        print(f"[guard] TIMEOUT after {seconds}s{(' in ' + label) if label else ''}"
              f" -- returning default", flush=True)
        return default
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_in_subprocess(fn, *args, seconds=30, default=None, **kwargs):
    """Last resort for a C-level hang SIGALRM cannot interrupt: run in a child
    process and kill it on overrun. Costs a fork per call, so reserve it for
    calls already known to hang (e.g. xtb), not for every molecule."""
    import multiprocessing as mp

    def _target(q):
        try:
            q.put(fn(*args, **kwargs))
        except Exception as e:
            q.put(("__error__", repr(e)))

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_target, args=(q,))
    p.start()
    p.join(seconds)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            p.kill()
        print(f"[guard] subprocess TIMEOUT after {seconds}s -- killed", flush=True)
        return default
    try:
        out = q.get_nowait()
    except Exception:
        return default
    if isinstance(out, tuple) and out and out[0] == "__error__":
        return default
    return out


def default_n_atoms(mol):
    """Heavy-and-hydrogen atom count of a chem.Molecule, up to the first STOP.

    Duck-typed on `.atoms` so this module stays independent of chem.py. A
    sequence terminates at the first STOP token (0), so the count is the index
    of that token rather than the length of the padded tensor.
    """
    a = mol.atoms
    stops = (a == 0).nonzero()
    if len(stops):
        return int(stops[0])
    return int(a.shape[0])


def guarded_reward(reward_fn, seconds=20, floor=-5.0, max_atoms=None,
                   n_atoms_of=default_n_atoms, report_every=25):
    """Wrap a reward function so no single molecule can stall the run.

    A molecule that overruns is scored at `floor` -- the same value already used
    for unparseable molecules, so the training signal treats "pathological
    geometry that breaks bond perception" the same way it treats "invalid",
    which is what it is.

    `max_atoms` short-circuits before the expensive call: bond perception cost
    grows sharply with size, and the largest samples are both the most likely to
    hang and the least likely to be useful.

    max_atoms IS NOT A FREE SAFETY MARGIN. Flooring every molecule above a size
    trains the policy to avoid that size, which changes the objective rather
    than merely guarding it. GEOM-sized molecules run to 80-100 heavy atoms
    against a max_len of 192, so a ceiling anywhere near the sampled range
    silently biases the reward distribution. It defaults to off, and should stay
    off unless a stack dump has actually shown large molecules to be the cause.
    """
    stats = {"timeouts": 0, "oversize": 0, "calls": 0}

    def wrapped(mol):
        stats["calls"] += 1
        if max_atoms is not None and n_atoms_of is not None:
            try:
                if n_atoms_of(mol) > max_atoms:
                    stats["oversize"] += 1
                    return floor
            except Exception:
                pass
        out = call_with_timeout(reward_fn, mol, seconds=seconds, default=None,
                                label="reward_fn")
        if out is None:
            stats["timeouts"] += 1
            # a guard that fires constantly is reshaping the reward, not
            # protecting it -- make that visible in the log rather than leaving
            # it to be inferred from a depressed mean
            if report_every and stats["timeouts"] % report_every == 0:
                print(f"[guard] {stats['timeouts']} timeouts / {stats['calls']} "
                      f"reward calls ({100*stats['timeouts']/stats['calls']:.1f}%)"
                      f" -- raise --guard_reward_timeout if this is not ~0",
                      flush=True)
            return floor
        return out

    wrapped.stats = stats
    return wrapped


# ------------------------------------------------------------------ layer 3

class StallGuard:
    """Watchdog thread: if no heartbeat arrives for `timeout_s`, dump every
    thread's stack and exit with STALL_EXIT_CODE.

    Exiting is deliberate. A hung process holds the GPU indefinitely and
    produces nothing; a supervisor that restarts from the last checkpoint loses
    minutes instead of hours. `flush_fn` runs first so the molecule log and the
    shard buffer are written before the process goes away.

    Under multi-device training each rank runs its own guard, and one rank
    exiting takes the job down. That is the intended behaviour: the supervisor
    restarts the whole job from the last checkpoint.
    """

    def __init__(self, timeout_s=1800, flush_fn=None, log_dir=None,
                 exit_on_stall=True):
        self.timeout_s = float(timeout_s)
        self.flush_fn = flush_fn
        self.log_dir = log_dir
        self.exit_on_stall = exit_on_stall
        self._last = time.time()
        self._note = "start"
        self._stop = threading.Event()
        self._thread = None

    def beat(self, note=""):
        self._last = time.time()
        if note:
            self._note = note

    def start(self):
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="stall-guard")
        self._thread.start()
        print(f"[guard] stall watchdog armed: {self.timeout_s/60:.1f} min")
        return self

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(min(30.0, self.timeout_s / 4)):
            idle = time.time() - self._last
            if idle < self.timeout_s:
                continue
            print(f"\n[guard] STALL: no progress for {idle/60:.1f} min "
                  f"(last: {self._note}). Dumping stacks.", flush=True)
            try:
                faulthandler.dump_traceback(all_threads=True)
                if self.log_dir:
                    os.makedirs(self.log_dir, exist_ok=True)
                    p = os.path.join(self.log_dir, f"stall-{int(time.time())}.log")
                    with open(p, "w") as f:
                        faulthandler.dump_traceback(file=f, all_threads=True)
                    print(f"[guard] stack dump -> {p}", flush=True)
            except Exception as e:
                print(f"[guard] dump failed: {e}", flush=True)
            if self.flush_fn is not None:
                try:
                    self.flush_fn()
                    print("[guard] flushed pending records", flush=True)
                except Exception as e:
                    print(f"[guard] flush failed: {e}", flush=True)
            sys.stdout.flush(); sys.stderr.flush()
            if self.exit_on_stall:
                # os._exit, not sys.exit: the main thread is blocked, so an
                # exception raised here would never reach it and the process
                # would stay hung holding the GPU
                os._exit(STALL_EXIT_CODE)
            return


def stall_guard_callback(timeout_s=1800, flush_fn=None, log_dir=None,
                         exit_on_stall=True):
    """Lightning callback wiring StallGuard to batch boundaries."""
    import lightning as L

    class _Cb(L.Callback):
        def __init__(self):
            self.guard = StallGuard(timeout_s, flush_fn, log_dir, exit_on_stall)

        def on_train_start(self, trainer, pl_module):
            if self.guard.flush_fn is None and getattr(pl_module, "recorder", None):
                self.guard.flush_fn = pl_module.recorder.flush
            self.guard.start()

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.guard.beat(f"epoch {trainer.current_epoch} batch {batch_idx} start")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            self.guard.beat(f"epoch {trainer.current_epoch} batch {batch_idx} done")

        # evaluation and the final dump can run far longer than a training
        # batch, so they beat too -- otherwise the watchdog fires on a run that
        # is working perfectly well, just slowly
        def on_validation_batch_end(self, trainer, pl_module, *a, **kw):
            self.guard.beat("validation")

        def on_validation_start(self, trainer, pl_module):
            self.guard.beat("validation start")

        def on_validation_end(self, trainer, pl_module):
            self.guard.beat("validation end")

        def on_train_epoch_end(self, trainer, pl_module):
            self.guard.beat(f"epoch {trainer.current_epoch} end")

        def on_train_end(self, trainer, pl_module):
            self.guard.stop()

    return _Cb()
