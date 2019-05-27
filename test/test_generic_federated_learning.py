"""
The following test checks if a test federated learning model works
"""

import torch
from torch import nn
from torch import optim
import syft as sy


def test_generic_federated_learning():
    """

    """
    hook = sy.TorchHook(torch)

    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")

    data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
    target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

    data_bob = data[0:2]
    target_bob = target[0:2]

    data_alice = data[2:]
    target_alice = target[2:]

    model = nn.Linear(2,1)
    opt = optim.SGD(params=model.parameters(),lr=0.1)

    data_bob = data_bob.send(bob)
    data_alice = data_alice.send(alice)
    target_bob = target_bob.send(bob)
    target_alice = target_alice.send(alice)
    datasets = [(data_bob,target_bob),(data_alice,target_alice)]

    def train():
        opt = optim.SGD(params=model.parameters(),lr=0.1)
        for _ in range(2):
            for data,target in datasets:
                model.send(data.location)
                opt.zero_grad()
                pred = model(data)
                loss = ((pred - target)**2).sum()
                loss.backward()
                opt.step()
                model.get()

    train()