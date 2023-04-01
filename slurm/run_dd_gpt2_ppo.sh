#!/bin/bash

WANDB_API_KEY=92a8247f01352422f27fab17382f1f897dd4f745  python scripts/training/train_text_generation.py \
--config_path scripts/training/task_configs/dialog/gpt2_ppo_edited.yml \
--experiment_name $1 \
--base_path_to_store_results /share/data/kartik-collab/geraldkwhite/ \
--log_to_wandb