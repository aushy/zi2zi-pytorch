#! /bin/sh

CURRENT=$(cd $(dirname $0);pwd)
cd $CURRENT

screen -dmS zi2zi python train.py --experiment_dir=experiment --gpu_ids=cuda:0 --batch_size=32 --epoch=100 --sample_steps=200 --checkpoint_steps=500 2>&1 | tee $(date "+%Y%m%d_%H%M")_log.txt
