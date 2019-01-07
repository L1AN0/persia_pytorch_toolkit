import torch
import numpy as np

def _iterate_over_container(inputs, func, instance_type=torch.Tensor):
    "Run a lambda over tensors in the container"
    def iterate(obj):
        if isinstance(obj, instance_type):
            return func(obj)
        if (isinstance(obj, tuple) or isinstance(obj, list)) and len(obj) > 0:
            return list(map(iterate, obj))
    # After iterate is called, a iterate cell will exist. This cell has a
    # reference to the actual function iterate, which has references to a
    # closure that has a reference to the iterate cell (because the fn is
    # recursive). To avoid this reference cycle, we set the function to None,
    # clearing the cell
    try:
        return iterate(inputs)
    finally:
        iterate = None


def to_device(container, device):
    """The device could be cuda:0 for example."""
    device = torch.device(device)
    return _iterate_over_container(container, lambda x: x.to(device))


def _numpy_dtype_to_torch_dtype(dtype: np.dtype):
    t = dtype.type
    if t is np.float64:
        return torch.float
    elif t is np.int64:
        return torch.long
    else:
        raise Exception("Unknown numpy dtype: " + str(t))


def to_device_from_numpy(container, device):
    """The device could be cuda:0 for example."""
    device = torch.device(device)
    return _iterate_over_container(container,
                                   lambda x: torch.as_tensor(x, dtype=_numpy_dtype_to_torch_dtype(x.dtype), device=device)
                                   np.ndarray
    )
