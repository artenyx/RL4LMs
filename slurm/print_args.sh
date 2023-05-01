#!/bin/bash

echo "SINGLE EXP ARGS:"
echo "1:task_name 2:base_model_name 3:ref_model_name 4:experiment_name 5:group_name"

echo "FULL EXP ARGS:"
echo "1:task_name 2:base_model_sm 3:run_sm_lg_exp 4:group_name"

echo "task_name: name of task in RL4LMs config"
echo "base_model_name: name of hf base model"
echo "ref_model_name: name of hf reference model"
echo "experiment_name: optional name of experiment if continuing (NONE if not used and need to specify gamma)"
echo "gamma: discount factor"
echo "base_model_sm: whether or not to use small base model (e.g. t5-small or distilgpt2)"
echo "run_sm_lg_exp: whether to run experiment where base model is larger than reference model"
echo "group_name: wandb init group name"
