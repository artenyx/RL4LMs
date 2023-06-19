from datetime import datetime
import yaml

'''
To Do: Create a function which writes to config file
with open(config_path, "r") as fp:
    config = yaml.safe_load(fp)

dt = datetime.now().strftime("%m%d%y%H%M%S")
config["wandb_id"] = dt if experiment_name is None else experiment_name[-12:]
config["wandb_group_id"] = group
config["alg"]["args"]["kl_type"] = kl_type
config["alg"]["args"]["off_policy"] = off_policy

base_model_str, ref_model_str = "", ""
if base_model_name is not None:
    config["alg"]["policy"]["args"]["model_name"] = base_model_name
    base_model_str = base_model_name.replace("-", "") + "base_"
if ref_model_name is not None:
    config["alg"]["policy"]["args"]["ref_model_name"] = ref_model_name
    ref_model_str = ref_model_name.replace("-", "") + "ref_"
if experiment_name is None:
    experiment_name = shorten_task_name(task_name) + base_model_str + ref_model_str + dt

if task_name == "imdb_text_continuation" and "imdb" not in base_model_name:
    config["tokenizer"]["model_name"] = "gpt2"
    
'''


