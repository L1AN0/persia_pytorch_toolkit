import torch


def _iterate_over_container(inputs, func):
    "Run a lambda over tensors in the container"
    def iterate(obj):
        if isinstance(obj, torch.Tensor):
            return func(obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(iterate, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(iterate, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(iterate, obj.items()))))
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
