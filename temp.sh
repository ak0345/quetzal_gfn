nohup: ignoring input
===================================================================
[run 1] sweep-nitrogen-tempgain-db-replay_on-b1
python gflow.py             --name sweep-nitrogen-tempgain-db-replay_on-b1             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 1             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 1] sweep-nitrogen-tempgain-db-replay_on-b1 (pid 72263, active=1)
===================================================================
[run 2] sweep-nitrogen-tempgain-db-replay_on-b10
python gflow.py             --name sweep-nitrogen-tempgain-db-replay_on-b10             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 10             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 2] sweep-nitrogen-tempgain-db-replay_on-b10 (pid 72338, active=2)
===================================================================
[run 3] sweep-nitrogen-tempgain-db-replay_off-b1
python gflow.py             --name sweep-nitrogen-tempgain-db-replay_off-b1             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 1             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain                          --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 3] sweep-nitrogen-tempgain-db-replay_off-b1 (pid 72477, active=3)
===================================================================
[run 4] sweep-nitrogen-tempgain-db-replay_off-b10
python gflow.py             --name sweep-nitrogen-tempgain-db-replay_off-b10             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 10             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain                          --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 4] sweep-nitrogen-tempgain-db-replay_off-b10 (pid 73646, active=3)
===================================================================
[run 5] sweep-nitrogen-tempgain-rtb-replay_on-b1
python gflow.py             --name sweep-nitrogen-tempgain-rtb-replay_on-b1             --quetzal_ckpt geom.ckpt             --objective rtb             --reward_beta 1             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 5] sweep-nitrogen-tempgain-rtb-replay_on-b1 (pid 74141, active=3)
===================================================================
[run 6] sweep-nitrogen-tempgain-rtb-replay_on-b10
python gflow.py             --name sweep-nitrogen-tempgain-rtb-replay_on-b10             --quetzal_ckpt geom.ckpt             --objective rtb             --reward_beta 10             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 6] sweep-nitrogen-tempgain-rtb-replay_on-b10 (pid 74587, active=3)
===================================================================
[run 7] sweep-nitrogen-tempgain-rtb-replay_off-b1
python gflow.py             --name sweep-nitrogen-tempgain-rtb-replay_off-b1             --quetzal_ckpt geom.ckpt             --objective rtb             --reward_beta 1             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain                          --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 7] sweep-nitrogen-tempgain-rtb-replay_off-b1 (pid 75019, active=3)
===================================================================
[run 8] sweep-nitrogen-tempgain-rtb-replay_off-b10
python gflow.py             --name sweep-nitrogen-tempgain-rtb-replay_off-b10             --quetzal_ckpt geom.ckpt             --objective rtb             --reward_beta 10             --reward nitrogen_count             --no_use_hidden_guide --use_prior_temp --use_residual_gain                          --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[warn] sweep-nitrogen-tempgain-rtb-replay_off-b1 exited with code 137 (see sweep_logs/sweep-nitrogen-tempgain-rtb-replay_off-b1.log)
[launch 8] sweep-nitrogen-tempgain-rtb-replay_off-b10 (pid 75526, active=3)
[warn] sweep-nitrogen-tempgain-rtb-replay_on-b1 exited with code 137 (see sweep_logs/sweep-nitrogen-tempgain-rtb-replay_on-b1.log)
[warn] sweep-nitrogen-tempgain-rtb-replay_on-b10 exited with code 137 (see sweep_logs/sweep-nitrogen-tempgain-rtb-replay_on-b10.log)
===================================================================
[run 9] sweep-osim-tempgain-db-replay_on-b1
python gflow.py             --name sweep-osim-tempgain-db-replay_on-b1             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 1             --reward guacamol --reward_smiles hard_osimertinib             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 9] sweep-osim-tempgain-db-replay_on-b1 (pid 75602, active=2)
===================================================================
[run 10] sweep-osim-tempgain-db-replay_on-b10
python gflow.py             --name sweep-osim-tempgain-db-replay_on-b10             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 10             --reward guacamol --reward_smiles hard_osimertinib             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 10] sweep-osim-tempgain-db-replay_on-b10 (pid 75740, active=3)
===================================================================
[run 11] sweep-osim-tempgain-db-replay_off-b1
python gflow.py             --name sweep-osim-tempgain-db-replay_off-b1             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 1             --reward guacamol --reward_smiles hard_osimertinib             --no_use_hidden_guide --use_prior_temp --use_residual_gain                          --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 11] sweep-osim-tempgain-db-replay_off-b1 (pid 76958, active=3)
===================================================================
[run 12] sweep-osim-tempgain-db-replay_off-b10
python gflow.py             --name sweep-osim-tempgain-db-replay_off-b10             --quetzal_ckpt geom.ckpt             --objective db             --reward_beta 10             --reward guacamol --reward_smiles hard_osimertinib             --no_use_hidden_guide --use_prior_temp --use_residual_gain                          --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 12] sweep-osim-tempgain-db-replay_off-b10 (pid 77377, active=3)
===================================================================
[run 13] sweep-osim-tempgain-rtb-replay_on-b1
python gflow.py             --name sweep-osim-tempgain-rtb-replay_on-b1             --quetzal_ckpt geom.ckpt             --objective rtb             --reward_beta 1             --reward guacamol --reward_smiles hard_osimertinib             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
[launch 13] sweep-osim-tempgain-rtb-replay_on-b1 (pid 77801, active=3)
===================================================================
[run 14] sweep-osim-tempgain-rtb-replay_on-b10
python gflow.py             --name sweep-osim-tempgain-rtb-replay_on-b10             --quetzal_ckpt geom.ckpt             --objective rtb             --reward_beta 10             --reward guacamol --reward_smiles hard_osimertinib             --no_use_hidden_guide --use_prior_temp --use_residual_gain             --use_replay --replay_fraction 0.25 --replay_strategy reward             --max_epochs 5             --steps_per_epoch 100             --eval_n 0             --final_n 0             --hist_every_n_epochs 0             --no_fcd_enabled             --no_eval_base
===================================================================
