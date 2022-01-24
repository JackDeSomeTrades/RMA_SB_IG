import yaml
import os
from box import Box
from cfg import CFGFILEPATH
from utils.helpers import get_project_root




taskname = "a1_task_rma_conf.yaml"


conf_file = os.path.join(CFGFILEPATH, taskname)

with open(conf_file, "r") as f:
    config = yaml.safe_load(f)
    cfg = Box(config)
    cfg = cfg.task_config

    asset_path = cfg.asset.file.format(ROOT_DIR=get_project_root())

            # print(child.attrib)
    print(robot)






