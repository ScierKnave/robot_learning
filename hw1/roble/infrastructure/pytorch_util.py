from typing import Union

import torch
from torch import nn

Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}
    
def build_mlp(
        input_size: int,
        output_size: int,
        **kwargs
    ):
    """
    Builds a feedforward neural network

    arguments:
    n_layers: number of hidden layers
    size: dimension of each hidden layer
    activation: activation of each hidden layer
    input_size: size of the input layer
    output_size: size of the output layer
    output_activation: activation of the output layer

    returns:
        MLP (nn.Module)
    """
    try:
        params = kwargs["params"]
    except:
        params = kwargs

    if isinstance(params["output_activation"], str):
        output_activation = _str_to_activation[params["output_activation"]]

    # TODO: return a MLP. This should be an instance of nn.Module
    layers = []
    _input_size = input_size
    for (_output_size, activation) in zip(kwargs["params"]["layer_sizes"], kwargs["params"]["activations"]):
        layers.append(nn.Linear(_input_size, _output_size))
        layers.append(_str_to_activation[activation])
        _input_size = _output_size
    layers.append(nn.Linear(_input_size, output_size))
    layers.append(output_activation)
    mlp = nn.Sequential(*layers)
    return mlp

    raise NotImplementedError

device = None

def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)

def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)

def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
