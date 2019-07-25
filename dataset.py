from typing import Sequence, TypeVar, Callable, List, NamedTuple, Optional, Tuple
import torch
from torch.utils.data import TensorDataset, Dataset
import numpy as np

class Trace(NamedTuple):
    time: np.ndarray
    state: np.ndarray
    communication: np.ndarray
    sensing: np.ndarray
    control: np.ndarray
    error: np.ndarray

class SequenceDataset(Dataset):
    def __init__(self, runs: Sequence[Tuple[torch.Tensor, ...]]) -> None:
        self._runs = runs

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return self._runs[index]

    def __len__(self) -> int:
        return len(self._runs)


def tensors_from_trace(trace: Trace) -> Tuple[torch.FloatTensor, ...]:
    return torch.FloatTensor(trace.sensing), torch.FloatTensor(trace.control)

def prepare(trace: Trace,  steps: Optional[int] = None, padding: bool = False,
            default_dt: float = 0.1
            ) -> Trace:
    if steps is not None:
        if len(trace.time) > steps:
            s = slice(steps)
            trace = Trace(*[x[s] for x in trace])
        elif len(trace.time) < steps:
            if padding:
                if len(trace.time) > 1:
                    dt = trace.time[-1] - trace.time[-2]
                else:
                    dt = default_dt
                items = [np.arange(0, dt * steps, dt)]
                # pad with (state, communication, sensing, 0, error)
                n = steps - len(trace.time)
                for data, k in zip(trace[1:], [1, 1, 1, 0, 1]):
                    last = [data[-1] * k for _ in range(n)]
                    items.append(np.concatenate([data, last]))
                trace = Trace(*items)
    return trace


def create_dataset(simulator, n_simulation, parameter = None, steps=2):
    # prepare data for the net
    net_inputs =[]
    net_outputs = []
    net_errors = []

    for i in range(n_simulation):
        states, target_vels, errors = simulator.run()
        net_inputs.append(states)
        net_outputs.append(target_vels)
        net_errors.append(errors)
    
    trace_len = n_simulation*net_inputs[0].shape[0]
    t_input = np.array(net_inputs).reshape(-1,net_inputs[0].shape[1],net_inputs[0].shape[2])
    t_output = np.array(net_outputs).reshape(-1,net_outputs[0].shape[1])
    t_errors = np.array(net_errors).reshape(-1)

    if parameter == 'com':
        traces=[]
        seq_length=trace_len//n_simulation
        for j in range(n_simulation):
            trace = Trace( np.arange(seq_length), t_input[j*seq_length:j*seq_length+seq_length,:,0], np.zeros(seq_length), t_input[j*seq_length:j*seq_length+seq_length,:,3:5],t_output[j*seq_length:j*seq_length+seq_length],t_errors[j*seq_length:j*seq_length+seq_length])
            traces.append(trace)
    else:
        trace = Trace(np.zeros(trace_len), t_input[:,:,0], np.zeros(trace_len), t_input[:,:,3:5],t_output,t_errors)
    
    if parameter == None:
        dataset = TensorDataset(torch.FloatTensor(trace.state), torch.FloatTensor(trace.control))
    elif parameter=='dis':
        ss = trace.sensing
        cs = trace.control
        dataset = TensorDataset(
            torch.FloatTensor(ss.reshape(ss.shape[0]*ss.shape[1], ss.shape[2])),
            torch.FloatTensor(cs.reshape(cs.shape[0]*ss.shape[1], 1))
            )
    elif parameter=='com':
        dataset = SequenceDataset([tensors_from_trace(prepare(trace, steps, padding= False)) for trace in traces])

    return dataset


