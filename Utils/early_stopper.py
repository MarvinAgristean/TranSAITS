import torch

class Early_stopper():
    """
    The Early Stopper
    Parameters
    ----------
    patience : patience
        number of epochs after which to stop without an improvement smaller than min_improvement
    min_improvement:
        minimal improvement that is accepted to not stop
    mode:
        which mode to optimize
    """

    def __init__(self, patience, min_improvement,mode = 'min'):
        self.patience = patience
        self.min_improvement = min_improvement
        self.mode = mode
        self.last_improvement_counter = 0
        if mode == 'min':
            self.last_metric = torch.inf

    def report(self, metric):
        """
        Parameters
        ----------
        metric :
            the metric value to track
        """
        if self.mode== 'min':
            if  self.last_metric - metric >= self.min_improvement:
                self.last_improvement_counter = 0
                self.last_metric = metric
            else:
                self.last_improvement_counter = self.last_improvement_counter +1

    def should_stop(self):
        """
        Returns
        -------
        Boolean:
            whether or not to stop, according to the reported metrics, patience and minimal improvement
        """
        return self.last_improvement_counter > self.patience