import yaml
import os

from cfg import CFGFILEPATH

taskname = "a1task_rma_conf.yaml"


conf_file = os.path.join(CFGFILEPATH, taskname)

with open(conf_file, "r") as f:
    config = yaml.safe_load(f)
    print(config)


