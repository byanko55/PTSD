from ops.misc import *

import sys
import json
import torch.nn as nn


_module_mapping = {
    'Conv1d': nn.Conv1d,
    'Conv2d': nn.Conv2d,
    'MaxPool1d': nn.MaxPool1d,
    'MaxPool2d': nn.MaxPool2d,
    'ReLU': nn.ReLU,
    'Dropout1d': nn.Dropout1d,
    'Dropout2d': nn.Dropout2d,
    'Linear': nn.Linear,
    'LogSoftmax': nn.LogSoftmax,
    'Flatten': nn.Flatten
}


def load_layer(name: str, params: dict) -> nn.Module:
    """
    Args:
        name (str): layer name.
        params (dict): kwargs to build a single neural layer.

    Returns:
        md (nn.Module): layer component defined in torch.nn.
    """

    if name not in _module_mapping:
        raise TypeError(" \
            [nnloader: load_layer] layer type %s does not exist \
        "%(name))

    try: 
        md = _module_mapping[name](**params)
    except TypeError:
        unknown_arg = str(sys.exc_info()[1]).split(' ')[-1]
        raise AttributeError(" \
            [nnloader: load_layer] layer type '%s' got an unexpected keyword argument %s \
        "%(name, unknown_arg))

    return md


def load_nn(model_file:Any) -> nn.Sequential:
    """
    Build a model by reading corresponding meta data file (json or pt).
    """

    if type(model_file) == str :
        # Model class must be defined somewhere
        with open(model_file) as f:
            md = json.load(f)

            ns = nn.Sequential()

            for lname, largs in md['nnet_seq'].items():
                md = load_layer(largs['type'], largs['params'])
                ns.add_module(lname, md)

            return ns
    
    return model_file


class CustomNN(nn.Module):
    def __init__(
            self, 
            model_file:str,
        ) -> None:
        """
        Basic model (i.e., couple of CNNs) to train a single dataset.
        Each argument (e.g., dim, #y_labels, etc) is subject to a specific dataset.

        Args
            model_file (str): a file-like object or a string containing a model file name.
        """

        super(CustomNN, self).__init__()

        # Read model structure from given json file
        with open(model_file) as f:
            md = json.load(f)

            ns = nn.Sequential()

            for lname, largs in md['nnet_seq'].items():
                md = load_layer(largs['type'], largs['params'])
                ns.add_module(lname, md)

            self.nn = ns

    def forward(self, input_data:torch.tensor) -> torch.tensor:
        """ 
        Inference.
        """
        
        y = self.nn(input_data)

        return y