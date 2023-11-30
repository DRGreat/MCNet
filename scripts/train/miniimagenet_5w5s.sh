python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -shot 5 -no_wandb -lr 0.01 -max_epoch 100 -depth 2 -num_heads 2

nohup python train.py -batch 128 -dataset miniimagenet -gpu 0 -extra_dir your_run -temperature_attn 5.0 -lamb 0.25 -shot 5 -no_wandb -lr 0.01 -max_epoch 100 -depth 2 -num_heads 2 &