import torch

def make_delta_simplified(missing_mask):
    """
    Parameters
    ----------
    missing_mask : torch tensor
        missing mask; 1 means the entry is seen, 0 means it is missing
    Returns
    -------
    delta: torch tensor
        containing the number of time steps in after the last seen value of the same feature for each entry
    """
    delta = torch.zeros_like(missing_mask)
    sequence_length = delta.shape[1]
    for t in range(1,sequence_length):
        delta[:,t,:][missing_mask[:,t-1,:] == 0] = delta[:,t-1,:][missing_mask[:,t-1,:] == 0] +1
        delta[:,t,:][missing_mask[:,t-1,:] == 1] =  1
    return delta
