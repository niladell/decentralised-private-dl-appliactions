import os
import torch
import numpy as np
from src.core import train_test_fn as tt
from tensorboardX import SummaryWriter
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
import xgboost as xgb

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_with_mi(model,
                  train_dataloader,
                  criterion,
                  optimizer,
                  device,
                #   attack_model,
                  out_dataloader,
                  attack_criterion,
                  attack_optimizer,
                  attack_batch_size = 128,
                  save_data = False,
                  mi_epochs=[],
                  start_epoch=0,
                  epochs=10,
                  log_iteration = 100,
                  logger_name='logs',
                  force_new_model = True,
                  mi_train_loader=None):
    logger.info('Starting training')
    assert mi_train_loader is not None, NotImplementedError('No MI trainset loader defined') # TODO GIVE AUTO OPTION

    train_scheduler = None
    if type(optimizer) == dict: # Is this the best way to send in optimizer wrappers?
        keys = optimizer.keys()
        if 'train_scheduler' in keys:
            train_scheduler = optimizer['train_scheduler']
    
        optimizer = optimizer['optimizer']
        if not issubclass(type(optimizer), torch.optim.Optimizer):
            raise TypeError('Optimizer from dict wrong type. Is' 
                            f'it may not be in the dictionary? \n{optimizer}')

    if type(mi_epochs) == int:
        mi_epochs = [mi_epochs]
    elif type(mi_epochs) == list:
        pass
    elif mi_epochs == 'all':
        mi_epochs = list(range(epochs))
    elif mi_epochs == 'last':
        mi_epochs = [len(epochs) - 1]
    else:
        raise NotImplementedError(f'mi_epochs not implemented for: {mi_epochs}')

    if (type(save_data) != str) and (save_data != False):
        save_data = 'mi_dataset'
    
    if logger_name[-1] != '/':
        logger_name += '/'
    
    os.makedirs(logger_name, exist_ok=True)
    _, folders, _ = next(os.walk(logger_name))
    if folders:
        run_number = len(folders)
    else:
        run_number = 1
        folders = ['run_001']
    base_log_folder = f'{logger_name}{folders[-1]}'
    checkpoint_f = f'{base_log_folder}/checkpoint'
    if os.path.exists(checkpoint_f):
        with open(checkpoint_f, 'r') as f:
            last_saved_epoch = int(f.read())
            logger.info('Checkpoint found: Epoch {last_saved_epoch}')
        if last_saved_epoch == (epochs - 1) or force_new_model:
            if type(force_new_model) == str or force_new_model == int:
                run_number = int(force_new_model)
                logger.info(f'Manually selected to run on run_{run_number:03d}')
            else:
                run_number += 1
                logger.info('Starting with a new clean model')

        else:
            model.load_state_dict(torch.load(f'{base_log_folder}/model_last.fl'))
    else:
        logger.info('No checkpoint file found, starting to train from scratch') 
        # with open(checkpoint_f, 'w') as f:
        #     f.write(str(0))   

    logger.info(f'Summary folder: "{logger_name}run_{run_number:03d}/"')
    summary_writer = SummaryWriter(logdir=f'{logger_name}run_{run_number:03d}/target') #logdir=f'logs/run_{run_number:03d}_{extra_naming_logger}/target')
    summary_writer_attack = SummaryWriter(logdir=f'{logger_name}run_{run_number:03d}/attack') #(logdir=f'logs/run_{run_number:03d}_{extra_naming_logger}/attack')

    def save_sample_images(writer, train_dataloader, out_dataloader):
        def _save_sample_imgs(writer, dataloader, tag, num_imgs=10):
            sample, _ = next(iter(dataloader))
            for i, img in enumerate(torch.unbind(sample)):
                if i >= num_imgs:
                    break
                im_max, im_min = torch.max(img), torch.min(img)
                img = (img - im_min)/(im_max - im_min)
                writer.add_image(tag, img, i)
        
        _save_sample_imgs(writer, train_dataloader, 'train_imgs')
        _save_sample_imgs(writer, out_dataloader, 'out(test)_imgs')
    save_sample_images(summary_writer, mi_train_loader, out_dataloader)

    # input_tensor = torch.Tensor(12, 3, 32, 32) #.to(device)
    # dummy_tensor_shape = [train_dataloader.batch_size] + list(train_dataloader.dataset[0][0].shape)
    # example_input_tensor = torch.Tensor()
    # summary_writer.add_graph(model, torch.autograd.Variable(example_input_tensor, requires_grad=True))


    # len_out = len(out_dataloader)*out_dataloader.batch_size
    # len_train = len(train_dataloader)*train_dataloader.batch_size
    # if 'mi_train_dataset' in extra_args:
    #     mi_train_dataset = extra_args['mi_train_dataset']
    # else:
    #     mi_train_dataset = train_dataloader.dataset 

    # if len_out < len_train:
    #     mi_train_sampler = torch.utils.data.SubsetRandomSampler([np.random.randint(0, len_train) for _ in range(len_out)])
    #     mi_train_loader = torch.utils.data.DataLoader(mi_train_dataset,
    #                                         batch_size=out_dataloader.batch_size, sampler=mi_train_sampler, num_workers=out_dataloader.num_workers)
    #     logger.info(f'Size of the (target) model training dataset {len(train_dataloader)*mi_train_loader.batch_size} (size of test dataset: {len(out_dataloader)*out_dataloader.batch_size})')
    #     logger.info(f'Size of MI train {len(mi_train_loader)*mi_train_loader.batch_size} and MI test {len(out_dataloader)*out_dataloader.batch_size}')
    # else:
    #     raise NotImplementedError('I did not expect the test set to be bigger than the train set.')

    # if save_data:
    #     if not os.path.exists(f'tmp/{save_data}'):
    #         os.makedirs(f'tmp/{save_data}', exist_ok=True)
    #     torch.save(attack_model.state_dict(), f'tmp/{save_data}/init_attack_model.fl')


    for current_epoch in range(start_epoch, epochs):
        if train_scheduler:
            train_scheduler.step(current_epoch)
        logger.info(f'Epoch {current_epoch}')
        for i, param_group in enumerate(optimizer.param_groups):
            logger.info(f'Current lr = {param_group["lr"]}')
            summary_writer.add_scalar(f'lr_{i}', param_group["lr"], current_epoch, time.time())

        # Train for a single epoch
        tt.train(model, train_dataloader, criterion, optimizer,
                device=device, start_epoch=current_epoch, epochs=current_epoch+1,
                log_iteration = log_iteration, summary_writer=summary_writer)
        tt.test(model, mi_train_loader, criterion=criterion, device=device,
                summary_writer=summary_writer, epoch=current_epoch, custom_tag='eval_train')
        tt.test(model, out_dataloader, criterion=criterion, device=device,
                summary_writer=summary_writer, epoch=current_epoch)

        torch.save(model.state_dict(), f'{base_log_folder}/model_last.fl')
        with open(checkpoint_f, 'w') as f:
            f.write(str(current_epoch))

        if current_epoch in mi_epochs:
            new_data = generate_mi_samples(model, mi_train_loader, out_dataloader, device=device, normalization_fn=torch.nn.Softmax(dim=1))
            order = lambda x: (np.flip(np.sort(x[0])).copy(), x[1])
            new_data = list(map(order, new_data))
            train_portion = 0.8 # TODO Make this an argument to be passed
            dataset_len = len(new_data)
            train_split_len = int(dataset_len * train_portion)
            mi_train, mi_test = torch.utils.data.random_split(new_data, [train_split_len, dataset_len - train_split_len])
            run_alternative_attacks(mi_train, mi_test, summary_writer=summary_writer_attack, epoch=current_epoch)


        ###################################
            # if save_data:
            #     folders_save_data = '/'.join(save_data.split('/')[:-1])
            #     if not os.path.exists(folders_save_data):
            #         os.makedirs(folders_save_data, exist_ok=True)
            #     np.save(f'{save_data}-epoch_{current_epoch}', (mi_train, mi_test))

            # values_top = np.array(mi_train)
            # train_top = np.vstack(values_top[values_top[:,1] == 1.0, 0])
            # out_top = np.vstack(values_top[values_top[:,1] == 0.0, 0])

            # train_top = np.fliplr(np.sort(train_top))
            # out_top = np.fliplr(np.sort(out_top))

            # plt.figure()
            # plt.scatter(out_top[:, 0], out_top[:, 1], c='b')
            # plt.scatter(train_top[:, 0], train_top[:, 1], c='r')
            # # plt.show()
            # plt.savefig(f'tmp/{current_epoch}-data.png')
            # plt.close()


            # to_tensor = lambda data: [(torch.tensor(x), torch.tensor(y).float()) for (x,y) in data]
            # mi_train, mi_test = to_tensor(mi_train), to_tensor(mi_test)
            # logger.info(f'Sample from training: {mi_train[:4]}')
            # logger.info(f'Sample from testing: {mi_test[:4]}')
            # mi_train = torch.utils.data.DataLoader(mi_train, batch_size=attack_batch_size, shuffle=True)
            # mi_test = torch.utils.data.DataLoader(mi_test, batch_size=attack_batch_size, shuffle=True)

            # attack_model.load_state_dict(torch.load(f'tmp/{save_data}/init_attack_model.fl'))
            # tt.train(attack_model, mi_train, attack_criterion, attack_optimizer, device=device, epochs=attack_epochs, summary_writer=summary_writer_attack)
            # tt.test(attack_model, mi_test, criterion=attack_criterion, device=device, summary_writer=summary_writer_attack, epoch=current_epoch)


def generate_mi_samples(model, train_dataloader, out_dataloader, device=None, normalization_fn=None):
    logger.info('Generating samples for MI')
    model.eval()
    data_collect = []
    membership_collect = []
    
    def _collect(model, dataloader, is_trainingdata: bool):
        if is_trainingdata:
            label_fn = np.ones
        else:
            label_fn = np.zeros
        
        for (sample, _) in dataloader:
            if device:
                sample = sample.to(device)
            posterior = model(sample.detach())
            if normalization_fn is None:
                if torch.max(posterior) > 1.0 or torch.min(posterior) < 0.0:
                    raise NotImplementedError('No normalization function given!')
            else:
                posterior = normalization_fn(posterior)
            
            data_collect.append(posterior.cpu().detach().numpy())
            membership_collect.append(label_fn(posterior.shape[0], dtype=np.float))
    _collect(model, train_dataloader, is_trainingdata=True)
    _collect(model, out_dataloader, is_trainingdata=False)

    # for i, ((sample_train, _), (sample_out, _)) in enumerate(zip(train_dataloader, out_dataloader)):
    #     if device:
    #         sample_train, sample_out = sample_train.to(device), sample_out.to(device)
    #     # TODO Now assuming it's already Sigmoid output
    #     post_train = model(sample_train.detach())
    #     post_out = model(sample_out.detach())
    #     if normalization_fn is None:
    #         if torch.max(post_train) > 1.0 or torch.min(post_train) < 0.0:
    #             raise NotImplementedError('No normalization function given!')
    #     else:
    #         post_train = normalization_fn(post_train)
    #         post_out = normalization_fn(post_out)

    #     data_collect.append(post_train.cpu().detach().numpy())
    #     data_collect.append(post_out.cpu().detach().numpy())
    #     membership_collect.append(np.ones(post_train.shape[0], dtype=np.float))
    #     membership_collect.append(np.zeros(post_out.shape[0], dtype=np.float))

    data_collect = np.concatenate(data_collect, axis=0)
    membership_collect = np.concatenate(membership_collect, axis=0)
    return [sample_member_pair for sample_member_pair in zip(data_collect, membership_collect)]

def run_alternative_attacks(mi_train, mi_test, summary_writer=None, epoch=0):
    logger.info(f'Running attacks (lin, xgb) -- Epoch {epoch}')
    X_train, y_train = list(zip(*mi_train))
    X_test, y_test = list(zip(*mi_test))

    X_train = np.vstack(X_train)
    X_test = np.vstack(X_test)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = linear_model.LogisticRegression(solver='lbfgs').fit(X_train, y_train)
    lin_train_score = clf.score(X_train, y_train)
    lin_test_score = clf.score(X_test, y_test)
#    print(f'Linear \t {lin_train_score} \t-- {lin_test_score}')
    
    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                    max_depth = 5, alpha = 10, n_estimators = 10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    predictions = [round(value) for value in y_pred]
    acc_train = metrics.accuracy_score(y_train, predictions)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    acc_test = metrics.accuracy_score(y_test, predictions)
    if summary_writer is not None:
        summary_writer.add_scalar('train/xgboost', acc_train, epoch, time.time())
        summary_writer.add_scalar('test/xgboost', acc_test, epoch, time.time())
        summary_writer.add_scalar('train/lin_fit', lin_train_score, epoch, time.time())
        summary_writer.add_scalar('test/lin_fit', lin_test_score, epoch, time.time())

#    print(f'XGBoost \t {acc_train} \t-- {acc_test}')
    logger.info(f'Attacks accuracy: Lin {lin_test_score}, XGB {acc_test}   (Epoch {epoch})')
    return [lin_train_score, lin_test_score, acc_train, acc_test]