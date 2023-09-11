import torch
# a primitive imputation method: using the last seen value for all missing values; 0 if no value has been seen before
# input: databatch X, missing mask of X, missing values already have 0 as value in X
# output: X imputated with the last seen value respectively or 0 if no value has been seen before


class last_imputation():

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
            imputed data; imputation by filling missing values with the last seen value
        """
        imputed_data = torch.clone(X)
        for i in range(X.shape[0]):
            for k in range(X.shape[2]):
                last_seen_value = 0
                for j in range(X.shape[1]):
                    if missing_mask[i,j,k] == 1:
                        last_seen_value = X[i,j,k]
                    else:
                        imputed_data[i,j,k] = last_seen_value
        return imputed_data
