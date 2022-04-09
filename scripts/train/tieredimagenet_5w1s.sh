python train.py -batch 256 -dataset tieredimagenet -gpu 1,0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -no_wandb -lr 0.01 -max_epoch 10 -milestones 4 6 8
