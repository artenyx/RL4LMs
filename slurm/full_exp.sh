#!/bin/bash

task_name="$1"
base_model_name="$2"

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

#setting reference model list
if [[ "$base_model_name" == gpt2 ]]; then
  ref_models=("gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
elif [[ "$base_model_name" == gpt2-small ]]; then
  ref_models=("gpt2-small" "gpt2" "gpt2-medium" "gpt2-large" "gpt2-xl")
elif [[ "$base_model_name" == t5-small ]]; then
  ref_models=("t5-small" "t5-base" "t5-large" "t5-3b" "t5-11b")
elif [[ "$base_model_name" == t5-base ]]; then
  ref_models=("t5-base" "t5-large" "t5-3b" "t5-11b")
fi

# setting gpu partition
if [[ "$task_name" == "common_gen" || "$task_name" == "summarization" ]]; then
  partition=contrib-gpu-long
else
  partition=speech-gpu
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
  for targ_kl in 2.2 2.4 2.6 2.8 3.0 3.2 # 4.0 5.0 6.0 7.0 8.0 # 0.6 0.8 1.0 1.2 # 1.4 1.6 1.8 2.0 #ce 4.0 5.0 6.0 7.0 8.0
  do
    sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $exp_arg NONE $group $kl_type $off_policy $exp $targ_kl"
    echo "sbatch arguments: $sbatch_params"
    sbatch $sbatch_params
  done
elif [[ "$exp" == targ_kl_ref_grid ]]; then
  #targ_kl_list=(2.2 2.4 2.6 2.8 3.0 3.2)
  targ_kl_list=(3.4 3.6 3.8 4.0 4.2 4.4)
  ref_models=("t5-3b")
  for ref_model_name in "${ref_models[@]}"
  do
    for targ_kl in "${targ_kl_list[@]}"
    do
      sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $ref_model_name NONE $group $kl_type $off_policy targ_kl $targ_kl"
      echo "sbatch arguments: $sbatch_params"
      sbatch $sbatch_params
    done
  done
elif [[ "$exp" == targ_kl_lr_grid ]]; then
  if [[ "$kl_type" == full_kl_2 ]]; then
    if [[ "$base_model_name" == gpt2 ]]; then
      lr_list=(0.0000007 0.0000008 0.0000009 0.000001)
      targ_kl_list=(0.5 0.8 1.0 1.4 1.8) #(0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4) #(0.5 0.8 1.0 1.2 1.4 1.6)
    else
      lr_list=(0.0000005 0.000001 0.0000015 0.000002 0.0000025)
      targ_kl_list=(0.2 0.6 0.8 1.0 1.2) #(0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4) #(0.5 0.8 1.0 1.2 1.4 1.6)
    fi
  elif [[ "$kl_type" == cross_entropy ]]; then
    lr_list=(0.000002 0.000003 0.000004) #(0.0000006 0.0000007 0.0000008 0.0000009 0.000001)
    targ_kl_list=(4.5) # 3.0 4.0 4.5 5.0 5.5 6.0
  elif [[ "$kl_type" == cross_entropy_2 ]]; then
    lr_list=(0.000002 0.000003 0.000004 0.000005) #(0.0000006 0.0000007 0.0000008 0.0000009 0.000001)
    targ_kl_list=(0.45) # 3.0 4.0 4.5 5.0 5.5 6.0
  fi
  for lr in "${lr_list[@]}"
  do
    for targ_kl in "${targ_kl_list[@]}"
    do
      sbatch_params="-p $partition -C 48g slurm/single_exp.sh $task_name $base_model_name $exp_arg NONE $group $kl_type $off_policy targ_kl,lr $targ_kl,$lr"
      echo "sbatch arguments: $sbatch_params"
      sbatch $sbatch_params
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
