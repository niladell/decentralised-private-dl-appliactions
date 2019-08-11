import torch
import torch.nn.functional as func
# import syft as sy
import logging
import time
from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


def train(model, dataloader, criterion, optimizer, device=None, epochs=10, log_iteration = 100, start_epoch=0, summary_writer = None):
    for epoch in range(start_epoch, epochs):

        running_loss = 0.0
        running_acc = 0.0
        steps = 0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if device:
                inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = outputs.argmax(dim=1)
            running_acc += predicted.eq(labels.long()).sum().item() / predicted.shape[0]

            running_loss += loss.item()
            steps += 1
            if log_iteration is not None:  # Is this the best way?
                if i % log_iteration == 0 or i == len(dataloader) - 1:
                    running_loss = running_loss / steps
                    running_acc = running_acc / steps
                    logger.info(f'[{epoch}, {i}] loss: {running_loss} -- acc: {running_acc}')

                    if summary_writer is not None:
                        summary_writer.add_scalar('train/loss', running_loss,
                                                    int(epoch*len(dataloader) + i), time.time())
                        summary_writer.add_scalar('train/accuracy', running_acc,
                                                    int(epoch*len(dataloader) + i), time.time())
                        last_layer = list(model.children())[-1]
                        for name, para in last_layer.named_parameters():
                            if 'weight' in name:
                                summary_writer.add_scalar('LastLayerGradients/grad_norm2_weights',
                                                          para.grad.norm(),
                                                          int(epoch*len(dataloader) + i),
                                                          time.time())
                            if 'bias' in name:
                                summary_writer.add_scalar('LastLayerGradients/grad_norm2_bias',
                                                          para.grad.norm(),
                                                          int(epoch*len(dataloader) + i),
                                                          time.time())
                    running_loss = 0.0
                    running_acc = 0.0
                    steps = 0


def train_federate_simple(model, dataloader, criterion, optimizer, device=None, epochs=10, log_iteration = 1000):
    """This function trains a model in a federate fashion. Despite that the training is performed
       in serie, meaning, the training is done one worker (user/party) at a time.

    Arguments:
        model {[type]} -- [description]
        dataloader {[type]} -- [description]
        criterion {[type]} -- [description]
        optimizer {[type]} -- [description]

    Keyword Arguments:
        epochs {int} -- [description] (default: {10})
        log_iteration {int} -- [description] (default: {1000})
    """
    # ASSERT that the model is Syft
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data
            if device:
                inputs, labels = inputs.to(device), labels.to(device)
            model.send(inputs.location)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            model.get()

            running_loss += loss.get().detach().cpu().numpy()
            if log_iteration is not None:  # Is this the best way?
                if i % log_iteration == log_iteration - 1:
                    logger.info('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / log_iteration))
                    running_loss = 0.0

    logger.info('Finished Training')


def accuracy_fn(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k) # ! This is taken out as we accumulate the count outside.mul_(100.0 / batch_size))
    return res


def test(model, dataloader, criterion=func.nll_loss, device=None, summary_writer=None, epoch=0, custom_tag='test', topk=(1,5,20)):
    model.eval()
    test_loss = 0.0
    # correct = 0

    acc = [0 for _ in topk]
    elements = 0 
    with torch.no_grad():
        for data, target in dataloader:
            if device:
                data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum of the batch loss
            batch_acc = accuracy_fn(output, target, topk)
            for i in range(len(topk)):
                acc[i] += int(batch_acc[i].item())
            # acc_1 += _acc_1
            # acc_5 += _acc_5
            # acc_20 += _acc_20

            # pred = output.argmax(1, keepdim=True) # get the index of the max log-probability
            # correct += pred.eq(target.long().view_as(pred)).sum().item()
            elements += output.shape[0]  # ? Are all batches same size? Can I bring this out?
    test_loss /= elements
    # accuracy = correct / elements
    correct = acc[0]
    for i in range(len(acc)):
        acc[i] /= elements
    # accuracy = acc_1 / elements
    # acc_5 = acc_5 / elements
    # acc_20 = acc_20 / elements

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f})\n'.format(
        test_loss, correct, elements, acc[0]))
    if summary_writer is not None:
        summary_writer.add_scalar(f'{custom_tag}/loss', test_loss,
                                    epoch, time.time())
        for k, val in zip(topk, acc):
            summary_writer.add_scalar(f'{custom_tag}/accuracy_{k}', val,
                                         epoch, time.time())
        # summary_writer.add_scalar(f'{custom_tag}/accuracy_5', acc_5,
        #                             epoch, time.time())
        # summary_writer.add_scalar(f'{custom_tag}/accuracy_20', acc_20,
        #                             epoch, time.time())
    return test_loss, acc

# def train_federate_simple(model, dataloader, criterion, optimizer, aggregate_method,
#                           epochs=10, workers_iterations=5, log_iteration = 1000):
#     """This function trains a model in a federate fashion. Training here is done in parallel
#     and then the results are aggregated

#     Arguments:
#         model {[type]} -- [description]
#         dataloader {[type]} -- [description]
#         criterion {[type]} -- [description]
#         optimizer {[type]} -- [description]

#     Keyword Arguments:
#         epochs {int} -- [description] (default: {10})
#         log_iteration {int} -- [description] (default: {1000})
#     """
#     # ASSERT that the model is Syft
#     for epoch in range(epochs):

#         workers_models = []
#         workers_optimizers = []
#         # TOOD workers = ALL (data.locations?)
#         for worker in workers:  # TODO Define workers? Can I do data.workers?
#             worker_model = model.copy().send(worker)
#             workers_models.append(worker_model)
#             # TODO What is optimizer here?
#             workers_optimizers(optim.SGD(params=worker_model.parameters(),
#                                          lr=learning_rate))

#         # ? TODO @async
#         for wi in range(workers_iterations):
            
#             for idx, w in enumerate(workers):
#                 workers_optimizers[idx].zero_grad()
#                 # ! TODO How to pass data
#                 worker_pred = workers_models[idx](worker_data??¿)
#                 worker_loss = criterion(worker)

#         running_loss = 0.0
#         for i, data in enumerate(dataloader, 0):
#             inputs, labels = data
#             model.send(data.location)

#             # zero the parameter gradients
#             optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             model.get()

#             running_loss += loss.item()
#             if log_iteration is not None:  # Is this the best way?
#                 if i % log_iteration == log_iteration - 1:
#                     logger.info('[%d, %5d] loss: %.3f' %
#                         (epoch + 1, i + 1, running_loss / log_iteration))
#                     running_loss = 0.0

#     logger.info('Finished Training')