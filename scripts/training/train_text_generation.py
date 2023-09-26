import os
from argparse import ArgumentParser
from datetime import datetime

import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
)

'''
TO DO:
- Change experiment name to wandb_id so that you can pass in an ID if you need to continue a run, but I think actual experiment name can be different
'''

param_registry = {
        "wandb_id": {"path": "wandb_id", "type": str},
        "wandb_group_id": {"path": "wandb_group_id", "type": str},
        "targ_kl": {"path": "alg.kl_div.target_kl", "type": float},
        "init_beta": {"path": "alg.kl_div.coeff", "type": float},
        "lr": {"path": "alg.args.learning_rate", "type": float},
        "ref_size": {"path": "alg.policy.args.ref_model_name", "type": str},
        "kl_type": {"path": "alg.args.kl_type", "type": str},
        "off_policy": {"path": "alg.args.off_policy", "type": bool},
        "base_model_name": {"path": "alg.policy.args.model_name", "type": str},
        "ref_model_name": {"path": "alg.policy.args.ref_model_name", "type": str},
        "tokenizer": {"path": "tokenizer.model_name", "type": str},
        "n_envs": {"path": "env.n_envs", "type": int},
    }

nenvs_registry = {
    "gpt2-xl": 10,
    "gpt2-large": 10,
    "gpt2-medium": 10,
    "gpt2": 10,
}


def update_config_parameter(config, param_key, param_value):
    param_path = param_registry[param_key]["path"]
    param_type = param_registry[param_key]["type"]
    keys = param_path.split(".")
    current_level = config
    for key in keys[:-1]:
        current_level = current_level.get(key, {})
    param_value = param_type(param_value)
    current_level[keys[-1]] = param_value


def update_config_for_experiment(config, update_params):
    dt, task_name, group, kl_type, off_policy, base_model_name, ref_model_name, sweep_parameter, sweep_value = update_params.values()

    update_config_parameter(config, "wandb_id", dt)
    if group is not None:
        update_config_parameter(config, "wandb_group_id", group)
    update_config_parameter(config, "kl_type", kl_type)
    update_config_parameter(config, "off_policy", off_policy)

    if sweep_parameter is not None:
        assert group is not None, "If performing a sweep, must have group name."
        delim = ","
        sweep_parameter = sweep_parameter.split(delim)
        sweep_value = sweep_value.split(delim)
        for param, val in zip(sweep_parameter, sweep_value):
            # update parameter value combination
            update_config_parameter(config, param, val)

    # custom parameters for full_kl_2 kl type
    if kl_type == "full_kl_2":
        if "targ_kl" not in sweep_parameter and "ref_size" not in sweep_parameter:
            best_targ_kl_registry = {"gpt2-xl": 1.2, "gpt2-large": 1.0}
            best_targ_kl = best_targ_kl_registry.get(ref_model_name, 1.2)
            update_config_parameter(config, "targ_kl", best_targ_kl)
        if "lr" not in sweep_parameter and "ref_size" not in sweep_parameter:
            best_lr_registry = {"gpt2-xl": 0.0000008, "gpt2-large": 0.0000007}
            best_lr = best_lr_registry.get(ref_model_name, 0.000001)
            update_config_parameter(config, "lr", best_lr)

    if base_model_name is not None:
        update_config_parameter(config, "base_model_name", base_model_name)
    if ref_model_name is not None:
        update_config_parameter(config, "ref_model_name", ref_model_name)
    if task_name == "imdb_text_continuation" and "imdb" not in base_model_name:
        update_config_parameter(config, "tokenizer", "gpt2")


def main(
        config_path: str,
        project_name: str,
        experiment_name: str,
        base_path_to_store_results: str,
        entity_name: str,
        log_to_wandb: bool,
        base_model_name: str,
        ref_model_name: str,
        task_name: str,
        group: str,
        kl_type: str,
        off_policy: bool,
        sweep_parameter: str,
        sweep_value: str,
):

    update_params = {
        "dt": datetime.now().strftime("%m%d%y%H%M%S%f"),
        "task_name": task_name,
        "group": group,
        "kl_type": kl_type,
        "off_policy": off_policy,
        "base_model_name": base_model_name,
        "ref_model_name": ref_model_name,
        "sweep_parameter": sweep_parameter,
        "sweep_value": sweep_value,
    }

    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    update_config_for_experiment(config, update_params)

    if group is None:
        experiment_name = config["wandb_id"]
    else:
        experiment_name = group + "_"
        if sweep_parameter is not None:
            experiment_name += str(sweep_value).replace("-", "") + "_" + config["wandb_id"]
        else:
            experiment_name += config["wandb_id"]

    # load tracker
    tracker = Tracker(
        base_path_to_store_results,
        config,
        project_name,
        experiment_name,
        entity_name,
        log_to_wandb,
    )

    # instantiate the trainer here
    if "supervised" in config["alg"]["id"]:
        trainer = SupervisedTrainer(
            tokenizer_config=config["tokenizer"],
            datapool_config=config["datapool"],
            alg_config=config["alg"],
            train_eval_config=config["train_evaluation"],
            tracker=tracker,
        )
    else:
        trainer = OnPolicyTrainer(
            tokenizer_config=config["tokenizer"],
            datapool_config=config["datapool"],
            reward_config=config["reward_fn"],
            env_config=config["env"],
            on_policy_alg_config=config["alg"],
            train_eval_config=config["train_evaluation"],
            tracker=tracker,
        )
    trainer.train_and_eval()


if __name__ == "__main__":
    parser = ArgumentParser(description="Fine-tune LM to generate controlled text")
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--project_name", type=str, help="WANDB project name", default="rl4lm_exps"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="WANDB experiment name",
        default=None,
    )
    parser.add_argument(
        "--entity_name", type=str, help="WANDB entity name", default=None
    )
    parser.add_argument(
        "--base_path_to_store_results",
        type=str,
        help="Base path to store experiment results",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        help="Base model hf name",
        default=None,
    )
    parser.add_argument(
        "--ref_model_name",
        type=str,
        help="Ref model hf name",
        default=None,
    )
    parser.add_argument(
        "--task_name",
        type=str,
        help="Task name e.g. daily dialogue",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--group",
        type=str,
        help="wandb group name",
        default=None,
    )
    parser.add_argument(
        "--kl_type",
        type=str,
        help="type of KL divergence in reward to use",
        default="standard",
    )
    parser.add_argument(
        "--off_policy",
        type=str,
        help="whether to do off policy learning or not",
        default="false",
        choices=["true", "false"]
    )
    parser.add_argument(
        "--sweep_parameter",
        type=str,
        help="if performing sweep, parameter name. if multiple params, separate with \",\" as delimiter.",
        default=None,
    )
    parser.add_argument(
        "--sweep_value",
        type=str,
        help="if performing sweep, parameter value. if multiple params, separate with \",\" as delimiter.",
        default=None,
    )

    args = parser.parse_args()
    args.off_policy = args.off_policy == "true"

    main(
        args.config_path,
        args.project_name,
        args.experiment_name,
        args.base_path_to_store_results,
        args.entity_name,
        args.log_to_wandb,
        args.base_model_name,
        args.ref_model_name,
        args.task_name,
        args.group,
        args.kl_type,
        args.off_policy,
        args.sweep_parameter,
        args.sweep_value,
    )

'''scratch code

class ParamPathRegistryDict(dict):
    def __init__(self, config, initial_dict=None):
        super().__init__()
        self.config = config
        if initial_dict:
            self.update(initial_dict)

    def __missing__(self, key):
        if key not in self.config:
            print(f"**KEY {key} CREATED**")
        return key


    task_name_registry = {
        "imdb_text_continuation": "imdb",
        "dialog": "dd",
        "common_gen": "cg",
        "summarization": "summ",
        "narrative_qa": "nqa",
        "iwslt2017": "iwslt",
        "human_judgement": "hj",
    }
    def shorten_task_name(task_name: str
                      ) -> str:
    task_name = task_name_registry[task_name] if task_name in list(task_name_registry) else task_name
    task_name += "_"
    return task_name


'''
