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

if [ -n "$7" ]; then
  exp_arg="$7"
else
  exp_arg=NONE
fi


# setting base model
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

# setting gpu partition
if [[ "$task_name" == common_gen ]]; then  partition=contrib-gpu-long
else
  partition=speech-gpu #contrib-gpu-long
fi

# setting experiment type
if [[ "$exp" == init_beta ]]; then
  for beta in 0.2 0.3 0.4 0.5
  do
    sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $exp_arg NONE $group $kl_type $off_policy $exp $beta"
    echo "sbatch arguments: $sbatch_params"
    sbatch $sbatch_params
  done
elif [[ "$exp" == targ_kl ]]; then
  for targ_kl in 0.5 0.8 1.0 1.2 1.4 1.6 1.8 2.0 # 4.0 5.0 6.0 7.0 8.0 # 0.6 0.8 1.0 1.2 # 1.4 1.6 1.8 2.0 #ce 4.0 5.0 6.0 7.0 8.0
  do
    sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $exp_arg NONE $group $kl_type $off_policy $exp $targ_kl"
    echo "sbatch arguments: $sbatch_params"
    sbatch $sbatch_params
  done
elif [[ "$exp" == targ_kl_lr_grid ]]; then
  for lr in 0.0000006 0.0000007 0.0000008 0.0000009 0.000001
  do
    for targ_kl in 0.5 0.8 1.0 1.2 1.4 1.6 # 4.0 5.0 6.0 7.0 8.0 # 0.6 0.8 1.0 1.2 # 1.4 1.6 1.8 2.0 #ce 4.0 5.0 6.0 7.0 8.0
    do
      sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $exp_arg NONE $group $kl_type $off_policy targ_kl,lr $targ_kl,$lr"
      echo "sbatch arguments: $sbatch_params"
      #sbatch $sbatch_params
    done
  done
elif [[ "$exp" == lr ]]; then
  for lr in 0.0000006 0.0000007 0.0000008 0.0000009 0.000001
  do
    sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $exp_arg NONE $group $kl_type $off_policy $exp $lr"
    echo "sbatch arguments: $sbatch_params"
    sbatch $sbatch_params
  done
elif [[ "$exp" == ref_size ]]; then
  for ref_model_name in ${ref_models[@]}
  do
    sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $ref_model_name NONE $group $kl_type $off_policy $exp $ref_model_name"
    echo "sbatch arguments: $sbatch_params"
    sbatch $sbatch_params
  done
fi
