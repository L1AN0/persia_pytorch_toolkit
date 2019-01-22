import torch
import os
from persia_pytorch_toolkit import utils

def _ensure_module(model):
    "handle DataParallel etc"
    if hasattr(model, 'module'):
        model = model.module
    return model


def checkpoint(model, directory, filename, add_time_str=True, extension=".pt"):
    model = _ensure_module(model)
    os.makedirs(directory, exist_ok=True)
    state_dict = model.state_dict()
    time_str = utils.current_time_str()
    torch.save(state_dict, os.path.join(directory, filename + "." + time_str + extension))


def get_model_size(model):
    """TODO: consider using nelements()"""
    model = _ensure_module(model)
    params = 0
    for p in model.parameters():
        tmp = 1
        for x in p.size():
            tmp *= x
        params += tmp
    return params


def flatten_model_and_grad_tensor(network, verbose=False):
    """
    caveats:
    1. can only be called after first loss.backward (so that grad variables are created)
    2. the tensors are all zero (so if you want to initialization plz reinitialize)
    """
    total_size = 0
    tensor_type = None
    for parameter in network.parameters():
        if tensor_type == None:
            tensor_type = parameter.data.type() # https://github.com/pytorch/pytorch/wiki/Breaking-Changes-from-Variable-and-Tensor-merge
        total_size += parameter.nelement()

    if verbose:
        print("total size: ", total_size)
        print("tensor type: ", tensor_type)

    tensor = torch.Tensor(total_size).type(tensor_type)
    storage = tensor.storage()
    grad_tensor = torch.Tensor(total_size).type(tensor_type)
    grad_storage = grad_tensor.storage()


    if verbose:
        print("create new continuous storage")

    current_offset = 0

    for parameter in network.parameters():
        backup = parameter.data.clone()
        parameter.data.set_(storage, current_offset, parameter.data.size())
        parameter.data.copy_(backup)
        parameter.grad.data.set_(grad_storage, current_offset, parameter.data.size())
        current_offset += parameter.data.nelement()
        print("parameter storage offset: ", parameter.data.storage_offset())

    if verbose:
        print("The C pointer: ", storage.data_ptr())
        print("The C pointer for grad: ", grad_storage.data_ptr())
        print("Is contiguous memory? ", tensor.is_contiguous())
        print("Is grad contiguous memory? ", grad_tensor.is_contiguous())

    return tensor, grad_tensor

def model_to_flatten_parameters(network):
    return torch._utils._flatten_dense_tensors(list(network.parameters()))

def model_to_flatten_gradients(network):
    return torch._utils._flatten_dense_tensors(list(map(lambda x: x.grad.data, network.parameters())))

def flatten_parameters_to_model(flatten_parameters, model):
    return torch._utils._unflatten_dense_tensors(flatten_parameters, model.parameters())
