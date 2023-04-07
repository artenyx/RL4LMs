import os
from argparse import ArgumentParser
from datetime import datetime

import yaml

from rl4lms.envs.text_generation.logging_utils import Tracker
from rl4lms.envs.text_generation.training_utils import (
    OnPolicyTrainer,
    SupervisedTrainer,
)

task_name_dict = {"imdb_text_continuation": "imdb_",
                  "dailydialogue": "dd_"}


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
):

    # load the config file
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)

    if experiment_name is None:
        if base_model_name is None:
            experiment_name = task_name_dict[task_name] + datetime.now().strftime("%m%d%y%H%M%S")
        else:
            experiment_name = task_name_dict[task_name] + base_model_name.replace("-", "") + "base_" + ref_model_name.replace("-", "") + "ref_" + datetime.now().strftime("%m%d%y%H%M%S")
            config["alg"]["policy"]["args"]["model_name"] = base_model_name
            config["alg"]["policy"]["args"]["ref_model_name"] = ref_model_name

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
        "--log_to_wandb", action="store_true", help="Whether to use wandb logging"
    )
    args = parser.parse_args()

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
    )
