#!/usr/bin bash
accelerate launch --config_file /public/xcx/Item/Pre-Train/CrabGPT/train_scripts/sft/accelerate_multi_gpus.yaml /public/xcx/Item/Pre-Train/CrabGPT/sft.py