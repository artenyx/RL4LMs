#!/bin/bash

task_name="$1"
base_model_name="$2"
ref_model_name="$3"
experiment_name="$4"
gamma="$5"

args=()
if [[ "$base_model_name" == *gpt2* ]]; then
  args+=("--config_path scripts/training/task_configs/${1}/gpt2_ppo.yml")
elif [[ "$base_model_name" == *t5* ]]; then
  args+=("--config_path scripts/training/task_configs/${1}/t5_ppo.yml")
fi

if [ -z "$4" ] || [ "$4" != NONE ]; then
  args+=("--experiment_name $experiment_name")
fi

if [ -z "$5" ] || [ "$4" != NONE ]; then
  args+=("--gamma $gamma")
fi

args+=('--base_path_to_store_results /share/data/kartik-collab/geraldkwhite/' \
'--log_to_wandb' \
"--base_model_name $base_model_name" \
"--ref_model_name $ref_model_name" \
"--task_name $task_name")

WANDB_API_KEY=92a8247f01352422f27fab17382f1f897dd4f745  python scripts/training/train_text_generation.py "${args[@]}"