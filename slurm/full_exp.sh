#!/bin/bash

task_name="$1"
base_model_sm="$2"
run_sm_lg_exp="$3"

if [ -n "$4" ]; then
  group="$4"
else
  group=NONE
fi

#setting base model
if [[ "$task_name" == imdb_text_continuation ]] || [[ "$task_name" == dialog ]]; then
  if [[ $base_model_sm == true ]]; then
    base_model_name=distilgpt2
  else
    base_model_name=gpt2
  fi
else
  if [[ $base_model_sm == true ]]; then
    base_model_name="t5-small"
  else
    base_model_name="t5-base"
  fi
fi

#setting reference model list
if [[ "$base_model_name" == *gpt2* ]]; then
  if [[ $run_sm_lg_exp == true ]]; then
    ref_models=("distilgpt2" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
  else
    ref_models=("distilgpt2" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
  fi
elif [[ "$base_model_name" == *t5* ]]; then
  if [[ $run_sm_lg_exp == true ]]; then
    ref_models=("t5-small" "t5-base" "t5-large" "t5-3b" "t5-11b")
  else
    ref_models=("t5-base" "t5-large" "t5-3b" "t5-11b")
  fi
fi

for ref_model_name in ${ref_models[@]}
do
  sbatch -p speech-gpu -C 48g slurm/single_exp.sh "$task_name" "$base_model_name" "$ref_model_name" NONE "$group"
done
