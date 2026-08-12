"""Times the ACTUAL dump scenario: rollout(2000) single batch, and rollout_chunked,
at diff_steps=18 (the real setting). This is what the 2-hour run actually did."""
import time, argparse, torch
ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True); a=ap.parse_args()
import gflow
from gflow import LitGFlowNet
ckpt=torch.load(a.ckpt,map_location="cpu",weights_only=False)
hp=ckpt.get("hyper_parameters",ckpt.get("hparams"))
lit=LitGFlowNet(hp); lit.load_state_dict(ckpt["state_dict"],strict=False)
lit=lit.to("cuda").eval()
lit.cfg.diff_steps=18  # the REAL setting; diff_steps=2 produces degenerate mols

print("testing rollout(500) single batch...")
t=time.time()
try:
    with torch.no_grad():
        out=lit.rollout(500,guide=None,sample_temp=lit.cfg.sample_temp,rand_eps=0.0,with_reward=False)
    print(f"  rollout(500): {time.time()-t:.1f}s  ({len(out['mols'])} mols, {(time.time()-t)/500*1000:.0f}ms/mol)")
except RuntimeError as e:
    print(f"  rollout(500) FAILED: {str(e)[:100]}")

print("testing rollout(2000) single batch (what the dump tries first)...")
t=time.time()
try:
    with torch.no_grad():
        out=lit.rollout(2000,guide=None,sample_temp=lit.cfg.sample_temp,rand_eps=0.0,with_reward=False)
    print(f"  rollout(2000): {time.time()-t:.1f}s  ({len(out['mols'])} mols)")
except RuntimeError as e:
    print(f"  rollout(2000) OOM/FAILED after {time.time()-t:.1f}s: {str(e)[:100]}")
    print("  -> dump would fall back to rollout_chunked here")

print("testing rollout_chunked(2000, chunk=500) -- the fallback path...")
t=time.time()
try:
    with torch.no_grad():
        out=lit.rollout_chunked(2000,guide=None,chunk=500,with_reward=False)
    print(f"  rollout_chunked(2000): {time.time()-t:.1f}s  ({len(out['mols'])} mols)")
except Exception as e:
    print(f"  rollout_chunked FAILED after {time.time()-t:.1f}s: {str(e)[:150]}")