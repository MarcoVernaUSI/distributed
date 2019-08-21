

from typing import List, Tuple, Sequence, Optional, TypeVar, Callable, NamedTuple, Optional
import torch
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm

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


class DistributedNet(torch.nn.Module):

    def __init__(self, input_size):
        super(DistributedNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 10)
        self.l2 = torch.nn.Linear(10, 1)

    def forward(self, xs):
        ys = F.torch.tanh(self.l1(xs))
        return self.l2(ys)

    def controller(self):
        def f(sensing):
            with torch.no_grad():
                return self(torch.FloatTensor(sensing)).numpy().flatten(),
        return f
