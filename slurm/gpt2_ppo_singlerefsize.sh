#!/bin/bash

WANDB_API_KEY=92a8247f01352422f27fab17382f1f897dd4f745  python scripts/training/train_text_generation.py \
--config_path "scripts/training/task_configs/${1}/gpt2_ppo.yml" \
--base_path_to_store_results /share/data/kartik-collab/geraldkwhite/ \
--log_to_wandb \
--base_model_name "$2" \
--ref_model_name "$3"