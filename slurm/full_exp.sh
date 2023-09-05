#!/bin/bash

task_name="$1"
base_model_sm="$2"

if [ -n "$3" ]; then
  group="$3"
else
  group=NONE
fi

if [ -n "$4" ]; then
  kl_type="$4"
else
  kl_type=NONE
fi

if [ -n "$5" ]; then
  off_policy="$5"
else
  off_policy=NONE
fi

if [ -n "$6" ]; then
  exp="$6"
else
  exp=NONE
fi

if [[ "$task_name" == imdb_text_continuation ]] || [[ "$task_name" == dialog ]] || [[ "$task_name" == human_judgement ]]; then
  if [[ $base_model_sm == true ]]; then
    base_model_name=distilgpt2
  else
    base_model_name=gpt2
  fi
else
  if [[ $base_model_sm == true ]]; then
    base_model_name=t5-small
  else
    base_model_name=t5-base
  fi
fi

#setting reference model list
if [[ "$base_model_name" == *gpt2* ]]; then
  ref_models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")

elif [[ "$base_model_name" == *t5* ]]; then
  ref_models=("t5-base" "t5-large" "t5-3b" "t5-11b")
fi

#setting beta experiments


#setting correct gpu partition
if [[ "$group" == *ENVS* ]] || [[ "$task_name" == common_gen ]]; then  partition=contrib-gpu-long
else
  partition=speech-gpu
fi

#partition=contrib-gpu-long
if [[ "$exp" == init_beta ]]; then
  for beta in 0.1 0.15 0.2 0.22 0.25 0.3
  do
    sbatch -p "$partition" -C 48g slurm/single_exp.sh "$task_name" "$base_model_name" "gpt2" NONE "$group" "$kl_type" "$off_policy" "$exp" "$beta"
  done
elif [[ "$exp" == targ_kl ]]; then
  for targ_kl in 4.0 5.0 6.0 7.0 8.0 # 0.6 0.8 1.0 1.2 # 1.4 1.6 1.8 2.0 #ce 4.0 5.0 6.0 7.0 8.0
  do
    sbatch -p "$partition" -C 48g slurm/single_exp.sh "$task_name" "$base_model_name" "gpt2-xl" NONE "$group" "$kl_type" "$off_policy" "$exp" "$targ_kl"
  done
elif [[ "$exp" == lr ]]; then
  for lr in 0.0000006 0.0000007 0.0000008 0.0000009 0.000001
  do
    sbatch -p "$partition" -C 48g slurm/single_exp.sh "$task_name" "$base_model_name" "gpt2-xl" NONE "$group" "$kl_type" "$off_policy" "$exp" "$lr"
  done
elif [[ "$exp" == ref_size ]]; then
  for ref_model_name in ${ref_models[@]}
  do
    sbatch -p "$partition" -C 48g slurm/single_exp.sh "$task_name" "$base_model_name" "$ref_model_name" NONE "$group" "$kl_type" "$off_policy" "$exp" "$ref_model_name"
    # bash slurm/single_exp.sh "$task_name" "$base_model_name" "$ref_model_name" NONE "$group" "$kl_type" "$off_policy" # for testing
  done
fi
# speech-gpu
# contrib-gpu-long