import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from Models.BRITS.brits import FeatureRegression
from Utils.metric_service import masked_mae_cal, masked_rmse_cal

# see in mrnn.py github repository of the SAITS paper https://github.com/WenjieDu/SAITS

class FCN_Regression(nn.Module):
    def __init__(self, feature_num, rnn_hid_size):
        super(FCN_Regression, self).__init__()
        self.feat_reg = FeatureRegression(rnn_hid_size * 2)
        self.U = Parameter(torch.Tensor(feature_num, feature_num))
        self.V1 = Parameter(torch.Tensor(feature_num, feature_num))
        self.V2 = Parameter(torch.Tensor(feature_num, feature_num))
        self.beta = Parameter(torch.Tensor(feature_num))  # bias beta
        self.final_linear = nn.Linear(feature_num, feature_num)

        m = torch.ones(feature_num, feature_num) - torch.eye(feature_num, feature_num)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.U.size(0))
        self.U.data.uniform_(-stdv, stdv)
        self.V1.data.uniform_(-stdv, stdv)
        self.V2.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, x_t, m_t, target):
        h_t = F.tanh(
            F.linear(x_t, self.U * self.m) +
            F.linear(target, self.V1 * self.m) +
            F.linear(m_t, self.V2) +
            self.beta
        )
        x_hat_t = self.final_linear(h_t)
        return x_hat_t


class MRNN(nn.Module):
    """ MRNN Model

       Parameters
       ----------
       seq_len:
           number of timestamps of the time series this model should impute
       feature_num:
           number features of the timeseries this model should impute
       rnn_hidden_size:
           hidden dimension of the RITS model
        device:
           which device should be used for the computation
       """
    def __init__(self, seq_len, feature_num, rnn_hidden_size, **kwargs):
        super(MRNN, self).__init__()
        # data settings
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.rnn_hidden_size = rnn_hidden_size
        self.device = kwargs['device']

        self.f_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.b_rnn = nn.GRUCell(self.feature_num * 3, self.rnn_hidden_size)
        self.rnn_cells = {'forward': self.f_rnn,
                          'backward': self.b_rnn}
        self.concated_hidden_project = nn.Linear(self.rnn_hidden_size * 2, self.feature_num)
        self.fcn_regression = FCN_Regression(feature_num, rnn_hidden_size)

    def gene_hidden_states(self, data, direction):
        values = data[direction]['X']
        masks = data[direction]['missing_mask']
        deltas = data[direction]['deltas']

        hidden_states_collector = []
        hidden_state = torch.zeros((values.size()[0], self.rnn_hidden_size), device=self.device)

        for t in range(self.seq_len):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]
            inputs = torch.cat([x, m, d], dim=1)
            hidden_state = self.rnn_cells[direction](inputs, hidden_state)
            hidden_states_collector.append(hidden_state)
        return hidden_states_collector

    def impute(self, data):
        """
            Parameters
            ----------
                inputs:
                    data: dictionary containing:
                        forward: dictionary containing:
                            X:  3 dimensional torch tensor
                                time series data that will be imputed; X[i,j,k] is the k.th feature measured at j.th time of the i.th
                                time series; missing data points are filled with 0
                            missing_mask: 3 dimensional torch tensor, same shape as X
                                missing mask of the data X; missing_mask[i,j,k] = 1, if the data point X[i,j,k] is seen and 0 else
                            deltas: 3 dimensional torch tensor, same shape as X
                                    timegaps of missing data, deltas[i,j,k] is the timegap to the last seen datapoint of feature k of the i.th time series;
                                    refer to the BRITS paper for more detailed information
                        backward: dictionary containing:
                            X:  3 dimensional torch tensor
                                time series data that will be imputed; X[i,j,k] is the k.th feature measured at (seq_len-j).th time of the i.th
                                time series; missing data points are filled with 0; reversed time order of forward['X']
                            missing_mask: 3 dimensional torch tensor, same shape as X
                                missing mask of the data X; missing_mask[i,j,k] = 1, if the data point X[i,j,k] is seen and 0 else
                            deltas: 3 dimensional torch tensor, same shape as X
                                    timegaps of missing data, deltas[i,j,k] is the timegap to the last seen datapoint of feature k of the i.th time series;
                                    refer to the BRITS paper for more detailed information
                Returns
                -------
                imputed_data: 3 dimensional torch tensor, same shape as input X
                       the imputed data
                tuple containing:
                    estimations: 3 dimensional torch tensor, same shape as input X
                        estimated data of the model
                    reconstruction_loss:
                        reconstruction loss of seen data
                """
        hidden_states_f = self.gene_hidden_states(data, 'forward')
        hidden_states_b = self.gene_hidden_states(data, 'backward')[::-1]

        values = data['forward']['X']
        masks = data['forward']['missing_mask']

        reconstruction_loss = 0
        estimations = []
        rnn_est = [] # selfmade
        for i in range(self.seq_len):  # calculating estimation loss for times can obtain better results than once
            x = values[:, i, :]
            m = masks[:, i, :]
            h_f = hidden_states_f[i]
            h_b = hidden_states_b[i]
            h = torch.cat([h_f, h_b], dim=1)
            RNN_estimation = self.concated_hidden_project(h)  # x̃_t
            RNN_imputed_data = m * x + (1 - m) * RNN_estimation
            FCN_estimation = self.fcn_regression(x, m, RNN_imputed_data)  # FCN estimation is output extimation
            reconstruction_loss += masked_rmse_cal(FCN_estimation, x, m) + masked_rmse_cal(RNN_estimation, x, m)
            estimations.append(FCN_estimation.unsqueeze(dim=1))
            rnn_est.append(RNN_estimation.unsqueeze(dim = 1)) # selfmade

        estimations = torch.cat(estimations, dim=1)
        rnn_est = torch.cat(rnn_est, dim=1) # selfmade
        imputed_data = masks * values + (1 - masks) * estimations
        #return imputed_data, [estimations, reconstruction_loss]
        return imputed_data, [estimations, reconstruction_loss, rnn_est]


    def forward(self, data, stage):
        """
              Parameters
              ----------
                  inputs:
                      data: dictionary containing:
                          forward: dictionary containing:
                              X:  3 dimensional torch tensor
                                  time series data that will be imputed; X[i,j,k] is the k.th feature measured at j.th time of the i.th
                                  time series; missing data points are filled with 0
                              missing_mask: 3 dimensional torch tensor, same shape as X
                                  missing mask of the data X; missing_mask[i,j,k] = 1, if the data point X[i,j,k] is seen and 0 else
                              deltas: 3 dimensional torch tensor, same shape as X
                                   timegaps of missing data, deltas[i,j,k] is the timegap to the last seen datapoint of feature k of the i.th time series;
                                   refer to the BRITS paper for more detailed information
                          backward: dictionary containing:
                              X:  3 dimensional torch tensor
                                  time series data that will be imputed; X[i,j,k] is the k.th feature measured at (seq_len-j).th time of the i.th
                                  time series; missing data points are filled with 0; reversed time order of forward['X']
                              missing_mask: 3 dimensional torch tensor, same shape as X
                                  missing mask of the data X; missing_mask[i,j,k] = 1, if the data point X[i,j,k] is seen and 0 else
                              deltas: 3 dimensional torch tensor, same shape as X
                                   timegaps of missing data, deltas[i,j,k] is the timegap to the last seen datapoint of feature k of the i.th time series;
                                   refer to the BRITS paper for more detailed information
                      stage: string
                            the stage in which the forward is called
                  Returns
                  -------
                  ret: dictionary containing:
                        imputed_data: 3 dimensional torch tensor, same shape as input X
                            the imputed data
                        reconstruction_loss:
                            reconstruction loss of the seen data
                        reconstruction_MAE:
                            reconstruction mae of the seen data
                        imputation_MAE:
                            mae of the imputation of the artificially masked data
                        if X_holdout was a key of the input the following two are also added:
                        X_holdout: torch tensor, same shape as X
                            the indicated values that were given as an input
                        indicating_mask: torch tensor, same shape as X
                            the indicating mask that was given as an input

                  """

        values = data['forward']['X']
        masks = data['forward']['missing_mask']
        imputed_data, [estimations, reconstruction_loss, rnn_est] = self.impute(data) #,rnn_est selfmade
        reconstruction_loss /= self.seq_len
        reconstruction_MAE = masked_mae_cal(estimations.detach(), values, masks)

        if stage == 'val':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(imputed_data, data['X_holdout'], data['indicating_mask'])
        else:
            imputation_MAE = torch.tensor(0.0)

        ret_dict = {
            'reconstruction_loss': reconstruction_loss, 'reconstruction_MAE': reconstruction_MAE,
            'imputation_loss': imputation_MAE, 'imputation_MAE': imputation_MAE,
            'imputed_data': imputed_data,
        }
        # changed
        ret_dict['estimations'] = estimations
        ret_dict['rnn_est'] = rnn_est

        if 'X_holdout' in data:
            ret_dict['X_holdout'] = data['X_holdout']
            ret_dict['indicating_mask'] = data['indicating_mask']
        return ret_dict