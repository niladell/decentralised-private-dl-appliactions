import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import selene_sdk as sel

from src.selene import utils
from src.selene import train_model
sel.TrainModel = train_model.TrainModel
sel.utils.NonStrandSpecific = utils.NonStrandSpecific

print('Start')
dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "selene_manuscript/case1/1_train_and_evaluate/train_and_eval.yml")
print(f'Will use file at {config_path}')
configs = sel.utils.load_path(config_path, instantiate=False)
print('Done config!')
sel.utils.parse_configs_and_run(configs, lr=0.01)
print('Done ALL!')