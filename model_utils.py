import torch
import os
import utils


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
