nohup python train.py -batch 256 -dataset tieredimagenet -gpu 3 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -no_wandb -lr 0.01 -max_epoch 10 -milestones 4 6 8 &
nohup python train.py -batch 256 -dataset tieredimagenet -gpu 2 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -shot 5 -no_wandb -lr 0.01 -max_epoch 10 -milestones 4 6 8 &

