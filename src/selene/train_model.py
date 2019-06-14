"""
The objective of this file is to overwrite the functions of Selene
 (https://github.com/FunctionLab/selene) and make it compatible with
 Syft/Federated Training
"""
import os
import warnings
import logging
from time import strftime
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

import selene_sdk
from selene_sdk.utils import initialize_logger
from selene_sdk.utils import PerformanceMetrics
from selene_sdk.utils import load_model_from_state_dict
from selene_sdk.train_model import _metrics_logger

import syft as sy

hook = sy.TorchHook(torch)
logging.captureWarnings(True)
logger = logging.getLogger("selene")

USE_CUDA = True  # TODO Avoid global vars. Specially when in multiple files
                 # TODO Pass from YAML file

if not torch.cuda.is_available() and USE_CUDA:
        USE_CUDA = False
        logger.warning('Cuda unabailable, falling back to CPU')
if USE_CUDA:
    # TODO Quickhack. Actually need to fix the problem moving the model to CUDA
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = torch.device("cuda" if USE_CUDA else "cpu")

def eval_addon(self, f):
    def wrapper():
        f()
        self.evaluate()
        return None
    return wrapper

class TrainModel(selene_sdk.TrainModel):

    def __init__(self, *args, **kwargs):
        """
        Constructs a new `TrainModel` object.
        """
        if args:
            warnings.warn(f'Unexpected argument passed to model: {args}.'
                          'Possible overwrite on YAML config or passed from custom code.')
        kwargs['model'] = kwargs['model'].type(torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor)

        workers_log_msg = ''
        if 'workers_list' in kwargs:
            workers_list = [x.strip() for x in kwargs['workers_list'].split(',')]
            del kwargs['workers_list']
            assert type(workers_list[0]) == str, NotImplementedError('Workers can only be string as of now')
            self._create_virtual_workers(workers_list)
            workers_log_msg += 'Running the model in federate mode.'
            workers_log_msg += f' -- Clients are {self.workers_list}'
        else:
            # If there are no workers (i.e. federated scheme) then fallback to the original functions
            workers_log_msg += 'Running the model in vanilla mode. No federated schema used.'
            self.train = super().train
            self._get_batch = super()._get_batch
        super().__init__(*args, **kwargs)
        self.train_and_validate = eval_addon(self, self.train_and_validate)
        logger.info(workers_log_msg)  # Needs to be run after calling super()

    def train(self):  #pylint: disable=method-hidden
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.

        """
        self.model.train()
        self.sampler.set_mode("train")

        inputs, targets = self._get_batch()

        if self.use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        # TODO Change to this:
        # if device:
        #     inputs, labels = inputs.to(device), labels.to(device)

        # ! FEDERATE
        self.model._send(inputs.location) # Is this needed¿? -- normal .send() did not work

        # ? This is probably not needed
        # inputs = Variable(inputs)
        # targets = Variable(targets)
        self.optimizer.zero_grad()

        inputs = inputs.transpose(1, 2)
        predictions = self.model(inputs)
        loss = self.criterion(predictions, targets)

        loss.backward()
        self.optimizer.step()

        self.model.get()
        return loss.get().detach().cpu().numpy() # loss.item() # TODO < Check why not working

    def _get_batch(self):  #pylint: disable=method-hidden
            """
            Fetches a mini-batch of examples

            Returns
            -------
            tuple(numpy.ndarray, numpy.ndarray)
                A tuple containing the examples and targets.

            """

            t_i_sampling = time()
            batch_sequences, batch_targets = self.sampler.sample(
                batch_size=self.batch_size)
            # TODO Create dedicated sampler
            batch_sequences = torch.tensor(batch_sequences, requires_grad=True, dtype=torch.float32)
            batch_targets = torch.tensor(batch_targets, requires_grad=True, dtype=torch.float32)

            try:
                batch_sequences = batch_sequences.send(
                                        self.workers_list[self.current_worker])
                batch_targets = batch_targets.send(
                                        self.workers_list[self.current_worker])
            except AttributeError:
                raise('Somethign went wrong with sending the workers data')

            self.current_worker = (self.current_worker + 1) % len(self.workers_list)
            logger.debug(f'Current data from: {batch_sequences.location}') 
            t_f_sampling = time()
            logger.debug(
                ("[BATCH] Time to sample {0} examples: {1} s.").format(
                    self.batch_size,
                    t_f_sampling - t_i_sampling))
            return (batch_sequences, batch_targets)

    def _create_virtual_workers(self, workers_names):
        workers_list = [sy.VirtualWorker(hook, id=i) for i in workers_names]
        self.workers_list = workers_list
        self.current_worker = 0
