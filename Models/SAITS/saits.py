from Models.SAITS.positional_encoder import *
from Models.SAITS.encoder_layer import *
from Utils.metric_service import *
import torch.nn.functional as F
# see SA_models.py in github reporitory of the SAITS paper and the numerated equations in the SAITS paper
# https://github.com/WenjieDu/SAITS


class SAITS(nn.Module):
    """ SAITS Model

       Parameters
       ----------
       d_time:
           number of timesteps of the time series this model should impute
       d_feature:
           number features of the timeseries this model should impute
       d_model:
           hidden dimension frequently used to represent data processed between the steps
       d_inner:
           hidden dimension of feed forward layers
       d_k:
           dimension of key in attention
       d_v:
           dimension of value in attention
       dropout:
           dropoutrate
       n_groups:
           number of groups of inner layers
       n_group_inner_layers:
           number of inner layers in a group
       n_head:
           number of attentionheads
       param_sharing_between_group:
           whether parameters should be shared between groups
       MIT:
           whether to have the masked imputation task used in training
       input_with_mask:
           whether to concat input with mask
       diagonal_attention_mask:
           whether diagonal attention masking is used
       device:
           which device should be used
       """
    def __init__(self, d_time, d_feature, d_model, d_inner, d_k, d_v,
                 dropout=0, n_groups=5, n_group_inner_layers=1, n_head=8,
                 param_sharing_between_group=False, MIT=True, input_with_mask=True,
                 diagonal_attention_mask=True,
                 device='cpu'):
        super().__init__()
        self.n_groups = n_groups
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = input_with_mask
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_between_group = param_sharing_between_group
        self.MIT = MIT
        self.diagonal_attention_mask = diagonal_attention_mask
        self.device = device

        if self.param_sharing_between_group:
            # For between_group, only need to create 1 group and repeat n_groups times while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v,self.diagonal_attention_mask,
                             self.device, dropout = 0, attn_dropout=0)
                for _ in range(n_group_inner_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v,self.diagonal_attention_mask,self.device, dropout, 0)
                for _ in range(n_group_inner_layers)
            ])
        else:  # then inner_group，inner_group is the way used in ALBERT
            # For inner_group, only need to create n_groups layers
            # and repeat n_group_inner_layers times in each group while forwarding
            self.layer_stack_for_first_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, self.diagonal_attention_mask,self.device, dropout, 0)
                for _ in range(n_groups)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v,self.diagonal_attention_mask,self.device, dropout, 0)
                for _ in range(n_groups)
            ])

        self.dropout = nn.Dropout(p=dropout).to(device)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time).to(device)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model).to(device)
        self.reduce_dim_z = nn.Linear(d_model, d_feature).to(device)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model).to(device)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature).to(device)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature).to(device)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature).to(device)

    def impute(self, inputs):
        """
            Parameters
            ----------
            inputs: dictionary containing:
                X: 3 dimensional torch tensor
                   time series data that will be imputed; X[i,j,k] is the k.th feature measured at j.th time of the i.th
                   time series; missing data points are filled with 0
                missing_mask: 3 dimensional torch tensor, same shape as X
                    missing mask of the data X; missing_mask[i,j,k] = 1, if the data point X[i,j,k] is seen and 0 else
            Returns
            -------
            X_c: 3 dimensional torch tensor, same shape as input X
                the imputed data
            X_tilde_1: 3 dimensional torch tensor, same shape as X
                imputation and reconstruction from the first dmsa block
            X_tilde_2: 3 dimensional torch tensor, same shape as X
                imputation and reconstruction from the second dsma block
            X_tilde_3: 3 dimensional torch tensor, same shape as X
                imputation and reconstruction from whole the model
        """
        X, masks = inputs['X'].to(self.device), inputs['missing_mask'].to(self.device)
        # first DMSA block
        # eq 10
        input_X_for_first = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X_for_first = self.embedding_1(input_X_for_first)
        enc_output = self.dropout(self.position_enc(input_X_for_first))  # +p in eq 10; see positional_encoder.py
        # eq 11
        if self.param_sharing_between_group:
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_first_block:
                    enc_output, _ = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_first_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, _ = encoder_layer(enc_output)
        # eq 12
        X_tilde_1 = self.reduce_dim_z(enc_output)
        # eq 13
        X_prime = masks * X + (1 - masks) * X_tilde_1

        # second DMSA block
        # eq 14
        input_X_for_second = torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime
        input_X_for_second = self.embedding_2(input_X_for_second)
        enc_output = self.position_enc(input_X_for_second)  # namely term alpha in math algo
        # eq 15
        if self.param_sharing_between_group:
            for _ in range(self.n_groups):
                for encoder_layer in self.layer_stack_for_second_block:
                    enc_output, attn_weights = encoder_layer(enc_output)
        else:
            for encoder_layer in self.layer_stack_for_second_block:
                for _ in range(self.n_group_inner_layers):
                    enc_output, attn_weights = encoder_layer(enc_output)
        # eq 16
        X_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(enc_output)))

        # attention-weighted combine
        # eq 17
        attn_weights = attn_weights.squeeze()  # namely term A_hat in math algo
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)
        # eq 18
        combining_weights = F.sigmoid(self.weight_combine(torch.cat([masks, attn_weights], dim=2)))  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        # eq 19
        X_tilde_3 = (1 - combining_weights) * X_tilde_2 + combining_weights * X_tilde_1
        # eq 20
        X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return X_c, [X_tilde_1, X_tilde_2, X_tilde_3]

    def forward(self, inputs, stage):
        """
            Parameters
            ----------
            inputs: dictionary containing:
                X: 3 dimensional torch tensor
                   time series data that will be imputed; X[i,j,k] is the k.th feature measured at j.th time of the i.th
                   time series; missing data points are filled with 0
                missing_mask: 3 dimensional torch tensor, same shape as X
                    missing mask of the data X; missing_mask[i,j,k] = 1, if the data point X[i,j,k] is seen and 0 else
                indicating_mask: 3 dimensional torch tensor, same shape as X
                    indicating mask of the data X; indicating_mask[i,j,k] = 1, if the data point X[i,j,k] was
                    artificially masked and 0 else
                X_holdout: 3 dimensional torch tensor, same shape as X
                    time series data of X including the artificially masked values
            stage: string
                'val' for validation stage, 'test' for test stage
            Returns
            -------
            dictionary containing:
                'imputed_data': torch tensor, same shape as input X
                    the imputed data
                'reconstruction_loss':
                    loss of the reconstruction task
                'imputation_loss':
                    loss of the imputation task
                'reconstruction_MAE':
                    MAE of the reconstruction task
                'imputation_MAE':
                    MAE of the imputation task
        """
        X, masks = inputs['X'].to(self.device), inputs['missing_mask'].to(self.device)
        reconstruction_loss = 0
        imputed_data, [X_tilde_1, X_tilde_2, X_tilde_3] = self.impute(inputs)

        reconstruction_loss += masked_mae_cal(X_tilde_1, X, masks)
        reconstruction_loss += masked_mae_cal(X_tilde_2, X, masks)
        final_reconstruction_MAE = masked_mae_cal(X_tilde_3, X, masks)
        reconstruction_loss += final_reconstruction_MAE
        reconstruction_loss /= 3

        #if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
        imputation_MAE = masked_mae_cal(X_tilde_3, inputs['X_holdout'], inputs['indicating_mask'])
        #else:
         #   imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_loss, 'imputation_loss': imputation_MAE,
                'reconstruction_MAE': final_reconstruction_MAE, 'imputation_MAE': imputation_MAE}
