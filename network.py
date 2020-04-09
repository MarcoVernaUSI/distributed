from typing import List, Tuple, Sequence, Optional

import torch
from torch.utils import data
import torch.nn.functional as F
from tqdm import tqdm
from dataset import Trace
import numpy as np
import numpy.ma as ma


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
              criterion_ = None,
              padded= False,
              ) -> Tuple[List[float], List[float]]:    
    dl = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)    
    tdl = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
   
    optimizer.zero_grad()
    if criterion_ is None:
        criterion = torch.nn.MSELoss()
    elif criterion_ == "bin":
        criterion = torch.nn.BCELoss()

    if training_loss is None:
        training_loss = []
    if testing_loss is None:
        testing_loss = []
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0

        for n, (inputs, labels) in enumerate(dl):
            output = net(inputs)

            if padded == True:
                losses = []
                for out, label in zip(output,labels):
                    label = unmask(label)
                    loss = criterion(out, label)
                    losses.append(loss)
                loss = torch.mean(torch.stack(losses))
            else:
                loss = criterion(output, labels)
            loss.backward()
            epoch_loss += float(loss)
            optimizer.step()
            optimizer.zero_grad()
        training_loss.append(epoch_loss)
        with torch.no_grad():
            if padded == True:
                test_losses =[]
                for t_inputs, t_labels in tdl:
                    t_output = net(t_inputs)
                    losses = []
                    for out, label in zip(t_output,t_labels):
                        label = unmask(label)
                        loss = criterion(out, label)
                        losses.append(loss)
                    loss = torch.mean(torch.stack(losses))
                    test_losses.append(float(loss))
                testing_loss.append(sum(test_losses))
            else:
                testing_loss.append(
                    sum([float(criterion(net(inputs), labels)) for inputs, labels in tdl]))

        print(epoch_loss)
    return training_loss, testing_loss


class CentralizedNet(torch.nn.Module):

    def __init__(self, N: int):
        super(CentralizedNet, self).__init__()
        # Per N 10
        self.l1 = torch.nn.Linear(N*2, 10)        # prima di 64 era 10
        self.l2 = torch.nn.Linear(10, N)

    def forward(self, ds):
        ys = F.torch.tanh(self.l1(ds))
        return self.l2(ys)

    def controller(self) -> Controller:
        def f(state: Sequence[State], sensing: Sequence[Sensing]) -> Tuple[Sequence[Control]]:            
            with torch.no_grad():
                #return self(torch.FloatTensor(state)).numpy(),
                return self(torch.FloatTensor(np.array(sensing).flatten().tolist())).numpy(),
        return f

class CentralizedNetL(torch.nn.Module):

    def __init__(self, N: int):
        super(CentralizedNetL, self).__init__()

        # Per N 10
        self.l1 = torch.nn.Linear(N, 64)        # prima di 64 era 10
        self.l2 = torch.nn.Linear(64, N)
        self.out = torch.nn.Sigmoid()

    def forward(self, ds):
        ys = F.torch.tanh(self.l1(ds))
        ys2 = self.out(self.l2(ys))
        return ys2

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

class DistributedNetL(torch.nn.Module):

    def __init__(self, input_size):
        super(DistributedNetL, self).__init__()
        self.l1 = torch.nn.Linear(input_size, 10)
        self.l2 = torch.nn.Linear(10, 1)
        self.out = torch.nn.Sigmoid()

    def forward(self, xs):
        ys = F.torch.tanh(self.l1(xs))
        ys2 = self.out(self.l2(ys))
        return ys2

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

def unmask(label):
        new = []
        for i in range(label.shape[0]):
            indices = np.where(label[i] < 0)
            new.append(np.delete(label[i], indices, axis=0))
        new = torch.stack(new)
        
        return new