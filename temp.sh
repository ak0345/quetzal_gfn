SUF="tempgain-ws-from-db-beta20-t1.0-e0.05-t1.0-e0.05"
ROOT="logs/quetzal-gfn"

python probe_tempgain.py \
  --guide_ckpts "$ROOT/gfn-geom-osim-comp0--$SUF/checkpoints/last.ckpt,$ROOT/gfn-geom-osim-comp2--$SUF/checkpoints/last.ckpt,$ROOT/gfn-geom-osim-comp3--$SUF/checkpoints/last.ckpt" \
  --guide_labels "c0,c2,c3" \
  --eval_rewards "gcomp:osimertinib:0=c0,gcomp:osimertinib:2=c2,gcomp:osimertinib:3=c3" \
  --weights "0.3333,0.3333,0.3334" \
  --route flow --train_betas "20,20,20" \
  --guide_source ema \
  --product_kind harmonic \
  --n_traj 400 \
  --n_valid 1000 \
  --out_dir "$ROOT/osim-compose-db-20-tempgain/tempgain-probe"