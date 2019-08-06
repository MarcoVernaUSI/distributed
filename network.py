from typing import List, Tuple, Sequence, Optional

import torch
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
from dataset import Trace

from typing import Sequence, TypeVar, Callable
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

def train_net(epochs: int,
              train_dataset: data.TensorDataset,
              test_dataset: data.TensorDataset,
              net: torch.nn.Module,
              batch_size: int = 100,
              learning_rate: float = 0.01,
              momentum: float = 0,
              training_loss: Optional[List[float]] = None,
              testing_loss: Optional[List[float]] = None,
              ) -> Tuple[List[float], List[float]]:    
    dl = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
    tdl = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
   
    optimizer.zero_grad()
    criterion = torch.nn.MSELoss()
    if training_loss is None:
        training_loss = []
    if testing_loss is None:
        testing_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0
        for n, (inputs, labels) in enumerate(dl):
            output = net(inputs)
            loss = criterion(output, labels)
            loss.backward()
            epoch_loss += float(loss)
            optimizer.step()
            optimizer.zero_grad()
        training_loss.append(epoch_loss)
        with torch.no_grad():
            testing_loss.append(
                sum([float(criterion(net(inputs), labels)) for inputs, labels in tdl]))
    return training_loss, testing_loss


class CentralizedNet(torch.nn.Module):

    def __init__(self, N: int):
        super(CentralizedNet, self).__init__()

        # Per N 10
        self.l1 = torch.nn.Linear(N, 128)        # prima di 64 era 10
        self.l2 = torch.nn.Linear(128, N)

    def forward(self, ds):
        ys = F.torch.tanh(self.l1(ds))
        return self.l2(ys)

    def controller(self) -> Controller:
        def f(state: Sequence[State], sensing: Sequence[Sensing]) -> Tuple[Sequence[Control]]:
            with torch.no_grad():
                return self(torch.FloatTensor(state)).numpy(),
        return f


class DistributedNet(torch.nn.Module):

    def __init__(self, input_size):
        super(DistributedNet, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 10)
        self.l2 = torch.nn.Linear(10, 1)

    def forward(self, xs):
        ys = F.torch.tanh(self.l1(xs))
        return self.l2(ys)

    def controller(self) -> Controller:
        def f(state: Sequence[State], sensing: Sequence[Sensing]) -> Tuple[Sequence[Control]]:
            with torch.no_grad():
                return self(torch.FloatTensor(sensing)).numpy().flatten(),
        return f


class DistributedNetLeft(DistributedNet):

    def __init__(self):
        super(DistributedNetLeft, self).__init__()
        self.l1 = torch.nn.Linear(1, 10)
        self.l2 = torch.nn.Linear(10, 1)

    def forward(self, xs):
        ys = F.torch.tanh(self.l1(xs[:, :1]))
        return self.l2(ys)
