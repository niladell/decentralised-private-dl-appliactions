import selene_sdk as sel

import train_model
import utils
sel.TrainModel = train_model.TrainModel
sel.utils.NonStrandSpecific = utils.NonStrandSpecific

print('Start')
configs = sel.utils.load_path("./manuscript/case1/1_train_and_evaluate/train_and_eval.yml", instantiate=False)
print('Done config!')
sel.utils.parse_configs_and_run(configs, lr=0.01)
print('Done ALL!')