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
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

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

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


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
        self.summary = SummaryWriter()
        # self.summary.add_graph(self.model)  # TODO Not working
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
        loss = self.criterion(predictions, targets.detach())  # TODO w/o .detach() crashes. Why?

        loss.backward()
        self.optimizer.step()

        self.model.get()
        loss = loss.get().detach().cpu().numpy() # loss.item() # TODO < Check why not working
        return loss

    def train_and_validate(self):
        """
        Trains the model and measures validation performance.

        """
        min_loss = self._min_loss
        scheduler = ReduceLROnPlateau(
            self.optimizer, 'max', patience=2, verbose=True,
            factor=0.8)

        time_per_step = []
        for step in tqdm(range(self._start_step, self.max_steps)):
            t_i = time()
            train_loss = self.train()
            t_f = time()
            time_per_step.append(t_f - t_i)
            self.summary.add_scalar('train/loss', train_loss, step)
            self.summary.add_scalar('train/lr', get_lr(self.optimizer), step)

            if step % self.nth_step_save_checkpoint == 0:
                checkpoint_dict = {
                    "step": step,
                    "arch": self.model.__class__.__name__,
                    "state_dict": self.model.state_dict(),
                    "min_loss": min_loss,
                    "optimizer": self.optimizer.state_dict()
                }
                if self.save_new_checkpoints is not None and \
                        self.save_new_checkpoints >= step:
                    checkpoint_filename = "checkpoint-{0}".format(
                        strftime("%m%d%H%M%S"))
                    self._save_checkpoint(
                        checkpoint_dict, False, filename=checkpoint_filename)
                    logger.debug("Saving checkpoint `{0}.pth.tar`".format(
                        checkpoint_filename))

                else:
                    self._save_checkpoint(
                        checkpoint_dict, False)

            # TODO: Should we have some way to report training stats without running validation?
            if step and step % self.nth_step_report_stats == 0:
                logger.info(("[STEP {0}] average number "
                             "of steps per second: {1:.1f}").format(
                    step, 1. / np.average(time_per_step)))
                time_per_step = []
                valid_scores = self.validate()
                validation_loss = valid_scores["loss"]
                self._train_logger.info(train_loss)
                to_log = [str(validation_loss)]
                for k in sorted(self._validation_metrics.metrics.keys()):
                    if k in valid_scores and valid_scores[k]:
                        to_log.append(str(valid_scores[k]))
                    else:
                        to_log.append("NA")
                self._validation_logger.info("\t".join(to_log))
                # scheduler.step(math.ceil(validation_loss * 1000.0) / 1000.0)
                scheduler.step(validation_loss)
                logger.debug(f'Scheduler up! -- Val loss> {validation_loss}')
                self.summary.add_scalar('validation/loss', validation_loss, step)
                self.summary.add_scalars('validation', self._validation_metrics)

                if validation_loss < min_loss:
                    min_loss = validation_loss
                    self._save_checkpoint({
                        "step": step,
                        "arch": self.model.__class__.__name__,
                        "state_dict": self.model.state_dict(),
                        "min_loss": min_loss,
                        "optimizer": self.optimizer.state_dict()}, True)
                    logger.debug("Updating `best_model.pth.tar`")
                logger.info("training loss: {0}".format(train_loss))
                logger.info("validation loss: {0}".format(validation_loss))

                # Logging training and validation on same line requires 2 parsers or more complex parser.
                # Separate logging of train/validate is just a grep for validation/train and then same parser.
        self.sampler.save_dataset_to_file("train", close_filehandle=True)
        logger.info('Running evaluation')
        self.evaluate()

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
