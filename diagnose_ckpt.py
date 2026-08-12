#!/usr/bin/env python3
"""Diagnose why the guide isn't loading in final_dump2.
Run: python diagnose_ckpt.py --ckpt logs/quetzal-gfn/<name>/checkpoints/last.ckpt"""
import argparse, torch
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True)
a = ap.parse_args()
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)

hp = ck.get("hyper_parameters", ck.get("hparams", None))
print("=== hyper_parameters present:", hp is not None)
if hp:
    # hp may be the dict directly, or nested under 'config'
    cfg = hp.get("config", hp) if isinstance(hp, dict) else hp
    for k in ("use_hidden_guide", "use_prior_temp", "use_residual_gain",
              "objective", "reward", "reward_benchmark", "name"):
        print(f"    {k} = {cfg.get(k) if isinstance(cfg, dict) else '??'}")

sd = ck["state_dict"]
guide_keys = [k for k in sd if k.startswith("guide") and "ema" not in k]
ema_keys   = [k for k in sd if k.startswith("guide_ema")]
print("\n=== guide.* keys in checkpoint (first 12):")
for k in guide_keys[:12]: print("   ", k, tuple(sd[k].shape))
print(f"   ... {len(guide_keys)} total guide.* keys, {len(ema_keys)} guide_ema.* keys")

# what does a freshly-rebuilt module EXPECT?
print("\n=== rebuilding module from hp to compare expected keys ===")
import gflow
from gflow import LitGFlowNet
try:
    lit = LitGFlowNet(hp if isinstance(hp, dict) else dict(hp))
    exp = [k for k in lit.state_dict() if k.startswith("guide") and "ema" not in k]
    print("   expected guide.* keys (first 12):")
    for k in exp[:12]: print("   ", k)
    missing = set(exp) - set(sd.keys())
    print(f"\n   EXPECTED-but-MISSING-from-ckpt: {len(missing)}")
    for k in list(missing)[:12]: print("     ", k)
    matched = set(exp) & set(sd.keys())
    print(f"   MATCHED: {len(matched)}/{len(exp)}")
except Exception as e:
    print("   rebuild failed:", e)