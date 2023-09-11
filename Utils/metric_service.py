import torch
# see utils.py in SAITS paper/implementation https://github.com/WenjieDu/SAITS


def masked_mae_cal(inputs, target, mask):
    """
    Parameters
    ----------
    inputs : torch tensor

    target: torch tensor
        same shape as input
    mask: torch tensor
        containing only ones and zeros; same shape as input
    Returns
    -------
    number
        the Mean Absolute Error as seen in the paper;
    """
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def masked_rmse_cal(inputs, target, mask):
    """
    Parameters
    ----------
    inputs : torch tensor

    target: torch tensor
        same shape as input
    mask: torch tensor
        containing only ones and zeros; same shape as input
    Returns
    -------
    number
        the Root Mean Square Error as seen in the paper;
    """
    return torch.sqrt(masked_mse_cal(inputs, target, mask))


def masked_mre_cal(inputs, target, mask):
    """
    Parameters
    ----------
    inputs : torch tensor

    target: torch tensor
        same shape as input
    mask: torch tensor
        containing only ones and zeros; same shape as input
    Returns
    -------
    number
        the Mean Relative Error as seen in the paper;
    """
    return torch.sum(torch.abs(inputs - target) * mask) / (torch.sum(torch.abs(target * mask)) + 1e-9)


def masked_mse_cal(inputs, target, mask):
    """
    Parameters
    ----------
    inputs : torch tensor

    target: torch tensor
        same shape as input
    mask: torch tensor
        containing only ones and zeros; same shape as input
    Returns
    -------
    number
        the Mean Square Error as seen in the paper;
    """
    return torch.sum(torch.square(inputs - target) * mask) / (torch.sum(mask) + 1e-9)


def evaluate_metrics(inputs, target, mask, should_print=False):
    """
    Parameters
    ----------
    inputs : torch tensor

    target: torch tensor
        same shape as input
    mask: torch tensor
        containing only ones and zeros; same shape as input
    should_print: Boolean
        whether or not the metrics shall be printed
    Returns
    -------
    dictionary:
        'mae':
            the Mean Absolute Error as seen in the SAITS paper
        'rmse':
            the Root Mean Square Error as seen in the SAITS paper;
        'mre':
            the Mean Relative Error as seen in the SAITS paper;
    """
    mae = masked_mae_cal(inputs, target, mask)
    mre = masked_mre_cal(inputs, target, mask)
    rmse = masked_rmse_cal(inputs, target, mask)
    if should_print:
        print('MAE: ', mae)
        print('MRE: ', mre)
        print('RMSE: ', rmse)
    return {
        'rmse': rmse,
        'mae': mae,
        'mre': mre
    }
