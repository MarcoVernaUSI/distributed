from enum import Enum
from random import shuffle
from typing import Sequence, Tuple, TypeVar, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

State = TypeVar('State')
Sensing = TypeVar('Sensing')
Control = TypeVar('Control')
Communication = TypeVar('Communication')
MState = Sequence[State]
MSensing = Sequence[Sensing]
MControl = Sequence[Control]

Dynamic = Callable[[Sequence[State], Sequence[Control]], MState]
Sensor = Callable[[MState], MSensing]
ControlOutput = Tuple[Sequence[Control], Sequence[Communication]]
Controller = Callable[[Sequence[State], Sequence[Sensing]], ControlOutput]

class Sync(Enum):
    random = 1
    sequential = 2
    sync = 3


class SNet(nn.Module):
    def __init__(self):
        super(SNet, self).__init__()
        self.l1 = nn.Linear(4, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


class SNetLeft(nn.Module):
    def __init__(self):
        super(SNetLeft, self).__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 2)

    def forward(self, input):
        ys = F.torch.tanh(self.l1(input))
        return self.l2(ys)


def input_from(ss, comm, i):
    return torch.cat((ss[i], comm[i:i+1], comm[i+2:i+3]), 0)


def input_from_left(ss, comm, i):
    return torch.cat((ss[i][:1], comm[i:i+1]), 0)


def init_comm(N: int):
    return Variable(torch.Tensor([0] * (N + 2)))


class ComNet(nn.Module):
    def __init__(self, N: int, sync: Sync = Sync.sequential, module: nn.Module = SNet,
                 input_fn=input_from) -> None:
        super(ComNet, self).__init__()
        self.single_net = module()
        self.N = N
        self.sync = sync
        self.input_fn = input_fn

    def step(self, xs, comm, sync: Sync):
        if sync == Sync.sync:
            input = torch.stack([self.input_fn(xs, comm, i) for i in range(self.N)], 0)
            output = self.single_net(input)
            control = output[:, 0]
            comm[1:-1] = output[:, 1]
        else:
            indices = list(range(self.N))
            if sync == Sync.random:
                shuffle(indices)
            cs = []
            for i in indices:
                output = self.single_net(self.input_fn(xs, comm, i))
                comm[i+1] = output[1]
                cs.append(output[:1])
            control = torch.cat(cs, 0)
        return control

    def forward(self, runs):
        rs = []
        for run in runs:
            comm = init_comm(self.N)
            controls = []
            for xs in run:
                controls.append(self.step(xs, comm, self.sync))
            rs.append(torch.stack(controls))
        return torch.stack(rs)

    def controller(self, sync: Sync = Sync.sequential) -> Controller:
        N = self.N
        comm = init_comm(N)

        def f(state: Sequence[State], sensing: Sequence[Sensing]
              ) -> Tuple[Sequence[Control], Sequence[float]]:
            with torch.no_grad():
                sensing = torch.FloatTensor(sensing)                
                control = self.step(sensing, comm, sync=sync).numpy()
                return control, comm[1:-1].clone().numpy().flatten()
        return f
