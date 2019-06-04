"""
The objective of this file is to overwrite the functions of Selene
 (https://github.com/FunctionLab/selene) and make it compatible with
 Syft/Federated Training
"""
import os
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
logger = logging.getLogger("selene")

torch.set_default_tensor_type(torch.FloatTensor) # TODO testing


def eval_addon(self, f):
    def wrapper():
        f()
        self.evaluate()
        return None
    return wrapper

class TrainModel(selene_sdk.TrainModel):

    def __init__(self,
                 model,
                 data_sampler,
                 loss_criterion,
                 optimizer_class,
                 optimizer_kwargs,
                 batch_size,
                 max_steps,
                 report_stats_every_n_steps,
                 output_dir,
                 save_checkpoint_every_n_steps=1000,
                 save_new_checkpoints_after_n_steps=None,
                 report_gt_feature_n_positives=10,
                 n_validation_samples=None,
                 n_test_samples=None,
                 cpu_n_threads=1,
                 use_cuda=False,
                 data_parallel=False,
                 logging_verbosity=2,
                 checkpoint_resume=None,
                 metrics=dict(roc_auc=roc_auc_score,
                              average_precision=average_precision_score)):
        """
        Constructs a new `TrainModel` object.
        """
        self.model = model.type(torch.FloatTensor)
        self.sampler = data_sampler
        self.criterion = loss_criterion
        self.optimizer = optimizer_class(
            self.model.parameters(), **optimizer_kwargs)

        self.batch_size = batch_size
        self.max_steps = max_steps
        self.nth_step_report_stats = report_stats_every_n_steps
        self.nth_step_save_checkpoint = None
        if not save_checkpoint_every_n_steps:
            self.nth_step_save_checkpoint = report_stats_every_n_steps
        else:
            self.nth_step_save_checkpoint = save_checkpoint_every_n_steps

        self.save_new_checkpoints = save_new_checkpoints_after_n_steps

        logger.info("Training parameters set: batch size {0}, "
                    "number of steps per 'epoch': {1}, "
                    "maximum number of steps: {2}".format(
                        self.batch_size,
                        self.nth_step_report_stats,
                        self.max_steps))

        # torch.set_num_threads(cpu_n_threads)  # TODO This is breaking with Syft

        self.use_cuda = use_cuda
        self.data_parallel = data_parallel

        if self.data_parallel:
            self.model = nn.DataParallel(model)
            logger.debug("Wrapped model in DataParallel")

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()
            logger.debug("Set modules to use CUDA")

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

        initialize_logger(
            os.path.join(self.output_dir, "{0}.log".format(__name__)),
            verbosity=logging_verbosity)

        self._create_validation_set(n_samples=n_validation_samples)
        self._validation_metrics = PerformanceMetrics(
            self.sampler.get_feature_from_index,
            report_gt_feature_n_positives=report_gt_feature_n_positives,
            metrics=metrics)

        if "test" in self.sampler.modes:
            self._test_data = None
            self._n_test_samples = n_test_samples
            self._test_metrics = PerformanceMetrics(
                self.sampler.get_feature_from_index,
                report_gt_feature_n_positives=report_gt_feature_n_positives,
                metrics=metrics)

        self._start_step = 0
        self._min_loss = float("inf") # TODO: Should this be set when it is used later? Would need to if we want to train model 2x in one run.
        if checkpoint_resume is not None:
            checkpoint = torch.load(
                checkpoint_resume,
                map_location=lambda storage, location: storage)

            self.model = load_model_from_state_dict(
                checkpoint["state_dict"], self.model)

            self._start_step = checkpoint["step"]
            if self._start_step >= self.max_steps:
                self.max_steps += self._start_step

            self._min_loss = checkpoint["min_loss"]
            self.optimizer.load_state_dict(
                checkpoint["optimizer"])
            if self.use_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

            logger.info(
                ("Resuming from checkpoint: step {0}, min loss {1}").format(
                    self._start_step, self._min_loss))

        self._train_logger = _metrics_logger(
                "{0}.train".format(__name__), self.output_dir)
        self._validation_logger = _metrics_logger(
                "{0}.validation".format(__name__), self.output_dir)

        self._train_logger.info("loss")
        # TODO: this makes the assumption that all models will report ROC AUC,
        # which is not the case.
        self._validation_logger.info("\t".join(["loss"] +
            sorted([x for x in self._validation_metrics.metrics.keys()])))
        
        self.train_and_validate = eval_addon(self, self.train_and_validate)

    def train(self):
        """
        Trains the model on a batch of data.

        Returns
        -------
        float
            The training loss.

        """
        # logger.debug('yup! It\'s training') # * Some brightness
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

    def _get_batch(self):
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

            # ! TEMPORAL WORKAROUND!
            try:
                batch_sequences = batch_sequences.send(
                                        self.workers_list[self.current_worker])
                batch_targets = batch_targets.send(
                                        self.workers_list[self.current_worker])
            except AttributeError:
                workers = [sy.VirtualWorker(hook, id=f'worker_{i}') for i in range(2)]
                # workers, _ = create_virtual_workers(2) # TODO Pass this from the outside
                self.define_workers(workers)
                batch_sequences = batch_sequences.send(
                                        self.workers_list[self.current_worker])
                batch_targets = batch_targets.send(
                                        self.workers_list[self.current_worker])
            self.current_worker = (self.current_worker + 1) % len(self.workers_list)
            logger.debug(f'Current data from: {batch_sequences.location}') 
            t_f_sampling = time()
            logger.debug(
                ("[BATCH] Time to sample {0} examples: {1} s.").format(
                    self.batch_size,
                    t_f_sampling - t_i_sampling))
            return (batch_sequences, batch_targets)

    def define_workers(self, workers_list):
        self.workers_list = workers_list
        self.current_worker = 0
