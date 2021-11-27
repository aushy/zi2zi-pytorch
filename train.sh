#! /bin/sh

CURRENT=$(cd $(dirname $0);pwd)

(cd $CURRENT && screen -dmS zi2zi sh -c 'python train.py --experiment_dir=experiment --gpu_ids=cuda:0 --batch_size=32 --epoch=400 --sample_steps=5000 --checkpoint_steps=5000 --input_nc=1 2> $(date '+%Y%m%d_%H%M')_log.txt')
