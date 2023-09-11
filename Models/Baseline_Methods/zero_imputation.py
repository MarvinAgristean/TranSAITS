import torch
# a primitive imputation method: using zero for all missing values
# input: databatch X, missing mask of X, missing values already have 0 as value in X
# output: X imputated with zero


class zero_imputation():

    def __init__(self):
        pass

    def impute(self,X,missing_mask):
        """
        Parameters
        ----------
        X : torch tensor
            the torch tensor holding our data; 3 dimensional
        missing_mask: torch tensor
            missing mask for X; 1 means data is seen, 0 means it's unknown; same shape as X
        Returns
        -------
        torch tensor
            imputed data; imputation by filling missing values with zero
        """
        imputed_data = torch.clone(X) * missing_mask
        return imputed_data
