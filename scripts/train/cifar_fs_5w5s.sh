python train.py -batch 64 -dataset cifar_fs -gpu 1 -extra_dir your_run -temperature_attn 5.0 -lamb 0.5 -shot 5 -no_wandb -lr 0.01 -max_epoch 10 -milestones 4 6 8

nohup python train.py -batch 64 -dataset cifar_fs -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.5 -shot 5 -no_wandb -lr 0.01 -max_epoch 10 -milestones 4 6 8 &
