#!/bin/bash

WANDB_API_KEY=92a8247f01352422f27fab17382f1f897dd4f745  python scripts/training/train_text_generation.py \
--experiment_name "EXP1" \
--log_to_wandb