import torch

class median_imputation():
    # imputes data with the median of the corresponding feature of the training set
    def __init__(self, num_features):
        self.num_features = num_features
        self.median = torch.zeros(num_features)

    def train(self,X,missing_mask):
        """
        Parameters
            ----------
          X : torch tensor
             the torch tensor holding our data; missing values are set to 0; 3 dimensional
         missing_mask: torch tensor
             missing mask for X; 1 means data is seen, 0 means it's unknown; same shape as X
        """
        for i in range(self.num_features):
            all_i_feature_values = X[:,:,i]
            all_i_feature_masks = missing_mask[:,:,i]
            all_i_seen_feature_values = all_i_feature_values[all_i_feature_masks.bool()]
            self.median[i] = torch.median(all_i_seen_feature_values).item()

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
            imputed data; imputation by using the timeserieswise and featurewise median of the training;
            0 if no datapoint of this feature is seen
        """
        imputed_data = torch.clone(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if missing_mask[i,j,k] == 0:
                        imputed_data[i,j,k] = self.median[k].item()
        return imputed_data