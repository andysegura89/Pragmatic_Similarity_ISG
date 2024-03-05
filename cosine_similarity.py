import torch

def _check_list_type(list_to_check):
    """
    returns a list or numpy array as a torch.Tensor object in order for it
    to be used in the cosine similarity formula. Works for python lists, numpy arrays or
    torch.Tensor objects.
    :param list_to_check: the list that will be converted if it is not a torch.Tensor object.
    :return: torch.Tensor object representing the list passed.
    """
    return list_to_check if isinstance(list_to_check, torch.Tensor) else torch.tensor(list_to_check)

def get_cosine_similarity(list_1, list_2):
    """
    Uses the cosine_similarity formula on two lists. Lists must be of the same size.
    Works on python lists, numpy arrays, and torch.Tensor objects.
    :param list_1: first list passed to the cosine similarity formula
    :param list_2: second list passed to the cosine similarity formula
    :return: a floating point number representing the cosine similarity.
    This is a value between 0 and 1. The closer to 1, the more similar.
    """
    tensor_1 = _check_list_type(list_1)
    tensor_2 = _check_list_type(list_2)
    cos_sim = torch.nn.functional.cosine_similarity(tensor_1, tensor_2, dim=0)
    return cos_sim.item() # take out floating point number from tensor
