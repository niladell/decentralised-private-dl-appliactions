import torch

def train(model, dataloader, criterion, optimizer, epochs=10, log_iteration = 1000):
    for epoch in range(epochs):

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if log_iteration is not None:  # Is this the best way?
                if i % log_iteration == log_iteration - 1:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / log_iteration))
                    running_loss = 0.0

    print('Finished Training')



def train_federate_simple(model, dataloader, criterion, optimizer, epochs=10, log_iteration = 1000):
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
            model.send(data.location)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            model.get()

            running_loss += loss.item()
            if log_iteration is not None:  # Is this the best way?
                if i % log_iteration == log_iteration - 1:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / log_iteration))
                    running_loss = 0.0

    print('Finished Training')


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
#                     print('[%d, %5d] loss: %.3f' %
#                         (epoch + 1, i + 1, running_loss / log_iteration))
#                     running_loss = 0.0

#     print('Finished Training')