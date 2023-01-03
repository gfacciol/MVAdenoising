import torch


def load_net(cls, path):
    """
    Loads a model from a checkpoint.
    Args:
    - cls: The class of the model to be loaded.
    - path: The path to the checkpoint.
    """

    map_location = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    data = torch.load(path, map_location=map_location)

    if 'inspectors' in data:
        raise Exception(
            'This function does not handle inspector classes. Please use `load_inspector_net` instead.')

    model = cls(**data['model_args'])
    model.load_state_dict(data['model'])

    return model


def load_inspector_net(cls_model, cls_inspector, path):
    """
    Loads an inspector for a model from a checkpoint.
    Args:
    - cls_model: The class of the model to be inspected.
    - cls_inspector: The class corresponding to the inspector.
    - path: The path to the checkpoint.
    """

    map_location = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    data = torch.load(path, map_location=map_location)

    if 'inspectors' not in data:
        raise Exception(
            'This function can only handle inspector classes. Please use `load_net` instead.')

    model = cls_model(**data['model_args'])
    model.load_state_dict(data['model'])

    inspector = cls_inspector(model)
    inspector.load_state_dict(data['inspectors'], strict=False)

    return inspector
