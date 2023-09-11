import torch
# an primitive imputation method: using mean for all missing values
# input: databatch X, missing mask of X, missing values already have 0 as value in X
# ouput: X imputated with the mean of the not-missing values of X point/timeserieswise and featurewise


class mean_imputation():

    def __init__(self, num_features):
        self.mean = torch.zeros(num_features)
        self.num_features = num_features

    def train(self,X, missing_mask):
        """
        Parameters
        ----------
        X : torch tensor
            the torch tensor holding our data; missing values are set to 0; 3 dimensional
        missing_mask: torch tensor
            missing mask for X; 1 means data is seen, 0 means it's unknown; same shape as X
        Returns
        -------
        torch tensor
            imputed data; imputation by using the timeserieswise and featurewise mean;
            0 if no datapoint of this feature is seen
        """
        num_real_add = torch.sum(missing_mask, [0,1])
        mean = torch.sum(X, [0,1], keepdim=True) / num_real_add
        for i in range(self.num_features):
            if num_real_add[i] == 0:
                mean[i] = 0
        self.mean = mean

    def impute(self,X,missing_mask):
        """
        Parameters
        ----------
        X : torch tensor
            the torch tensor holding our data; missing values are set to 0; 3 dimensional
        missing_mask: torch tensor
            missing mask for X; 1 means data is seen, 0 means it's unknown; same shape as X
        Returns
        -------
        torch tensor
            imputed data; imputation by using the timeserieswise and featurewise mean of the training;
            0 if no datapoint of this feature is seen
        """
        imputed_data = torch.clone(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if missing_mask[i,j,k] == 0:
                        imputed_data[i,j,k] = self.mean[0,0,k]
        return imputed_data