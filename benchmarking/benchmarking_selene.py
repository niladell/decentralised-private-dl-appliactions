import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import time
import logging

import selene_sdk as sel

from src.selene import utils
from src.selene import train_model
sel.TrainModel = train_model.TrainModel  # TODO move to __init__ file
sel.utils.NonStrandSpecific = utils.NonStrandSpecific

logger = logging.getLogger('Benchmarking')
logger.setLevel(logging.INFO)

logger.info('Start Selene benchmark')
dir_path = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(dir_path, "selene_manuscript/case1/1_train_and_evaluate/train_and_eval.yml")
print(f'Will use file at {config_path}')

configs = sel.utils.load_path(config_path, instantiate=False)

logger.info('--- Start training federated ---')
time_fl = time.time()
sel.utils.parse_configs_and_run(configs, lr=0.2)
time_fl = time.time() - time_fl
logger.info(f'Total time {time_fl}s')
logger.info('--- End training federated ---')

del configs['train_model'][2]['workers_list']
logger.info('--- Start training vanilla ---')
time_v = time.time()
sel.utils.parse_configs_and_run(configs, lr=0.01)
time_v = time.time() - time_v
logger.info(f'Total time {time_v}s')
logger.info('--- End training vanilla ---')

logger.info(f'Federated (simple)/Vanilla time: {time_fl/time_v}s')