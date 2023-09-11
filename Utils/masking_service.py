import torch

def make_indicating_mask(missing_mask, p):
    """
    Parameters
    ----------
    missing_mask : torch tensor
        the missing mask with entries 1 for seen values and 0 for unseen values;
    p :
        the percentage of artificially masking
    Returns
    -------
    torch tensor
        the indicating mask with random indicated values; 1 means the entry is artificially masked, 0 means it isn't
    """
    # 1 if the value is artificially masked, 0 else
    if(len(missing_mask.shape)>2):
        indicating_mask = torch.zeros_like(missing_mask)
        for i in range(0,missing_mask.shape[0],1):
            indicating_mask_i = make_indicating_mask(missing_mask[i], p)
            indicating_mask[i] = indicating_mask_i
        return indicating_mask
    # 1 means X has the data point, 0 means it doesn't
    given_data_mask = missing_mask == 1
    given_data_indices = torch.nonzero(given_data_mask)
    num_given_data = len(given_data_indices)
    num_random_masking = int((num_given_data * p)//1)
    indicating_indices = given_data_indices[torch.randperm(num_given_data)[:num_random_masking]]
    indicating_mask = torch.zeros_like(missing_mask)
    for x, y in indicating_indices.tolist():
        indicating_mask[x, y] = 1
    # *1 turns boolean to 0 (false) and 1 (true)
    return indicating_mask * 1

def make_missing_mask(X):
    """
    Parameters
    ----------
    X : torch tensor
        the torch tensor holding our data; missing values are marked by a 'NaN' value
    Returns
    -------
    torch tensor
        the missing mask; 1 means the entry is seen, 0 means it is missing
    """
    if (len(X.shape) > 2):
        missing_mask = torch.zeros_like(X)
        for i in range(0, X.shape[0], 1):
            missing_mask_i = make_missing_mask(X[i])
            missing_mask[i] = missing_mask_i
        return missing_mask
    # only nan != nan
    missing_mask = X == X
    # *1 turns boolean to 0 (false) and 1 (true)
    return (missing_mask * 1).type(torch.float32)

