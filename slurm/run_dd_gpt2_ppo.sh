#!/bin/bash

echo -n "ENTER EXP NAME: "
read -r EXP_NAME

WANDB_API_KEY=92a8247f01352422f27fab17382f1f897dd4f745  python scripts/training/train_text_generation.py \
--experiment_name "$EXP_NAME" \
--log_to_wandb