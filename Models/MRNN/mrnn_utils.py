import torch
from Models.MRNN.mrnn import MRNN
from Utils.early_stopper import Early_stopper
from Utils.metric_service import masked_mae_cal, masked_rmse_cal, masked_mre_cal
from Models.BRITS.delta_service import make_delta_simplified
from torch.utils.data import DataLoader



def mrnn_training_loop(loader, model, optimizer, data_set, missing_mask, indicating_mask, deltas, device):
    """
    Parameters
    ----------
    loader: DataLoader
        dataloader of the indice of the data_set, missing_mask and indicating_mask
    model: M-RNN
        M-RNN model that shall be trained here
    optimizer:
        the optimizer that shall be used in this training epoch
    data_set: torch tensor
        dataset used for training
    missing_mask: torch tensor
        missing mask of the data_set, already contains artifially masked data
    indicating_mask: torch tensor
        indicating mask of the data_set and missing_mask
    deltas:
        the deltas, according to the missing mask
    device:
        the device to run the calculation on
    Returns
    -------
    number:
        the mean loss of this epoch
    """
    model.train()
    mean_loss = 0
    counter = 0
    for indice in loader:
        counter = counter + 1
        data_batch = data_set[indice]
        missing_mask_batch = missing_mask[indice]
        indicating_mask_batch = indicating_mask[indice]
        deltas_batch = deltas[indice]
        data_batch_backwards = torch.flip(data_batch, [1])
        missing_mask_batch_backwards = torch.flip(missing_mask_batch, [1])
        deltas_batch_backwards = make_delta_simplified(missing_mask_batch_backwards).to(device)
        input = {  'X_holdout': data_batch * indicating_mask_batch,
                    'indicating_mask': indicating_mask_batch,
                    'forward': {'X': data_batch * missing_mask_batch,
                        'missing_mask': missing_mask_batch,
                       'deltas': deltas_batch},
                    'backward': {'X': data_batch_backwards*missing_mask_batch_backwards,
                        'missing_mask': missing_mask_batch_backwards,
                        'deltas': deltas_batch_backwards}}
        output = model.forward(input, 'val')
        outer_recon_loss = masked_rmse_cal(output['estimations'],data_batch*missing_mask_batch, missing_mask_batch) + masked_rmse_cal(output['rnn_est'],data_batch*missing_mask_batch, missing_mask_batch)

        #loss_2 = output['reconstruction_loss'] + output['imputation_loss'] # intended way, but nAn error
        loss = outer_recon_loss + output['imputation_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += loss
    return mean_loss/(counter+0.0001)

def mrnn_evaluate(model, data_set, missing_mask, indicating_mask, deltas):
    """
    Parameters
    ----------
    model: M-RNN
        M-RNN model that shall be trained here
    data_set: torch tensor
        dataset used for evalutation
    missing_mask: torch tensor
        missing mask of the data_set, already contains artifially masked data
    indicating_mask: torch tensor
        indicating mask of the data_set and missing_mask
    deltas:
        the delta tensor of the missing_mask
    Returns
    -------
    dictionary:
        'mae':
            the Mean Absolute Error of the models estimation on this data_set according to the indicating_mask
        'rmse':
            the Root Mean Square Error of the models estimation on this data_set according to the indicating_mask;
        'mre':
            the Mean Relative Error of the models estimation on this data_set according to the indicating_mask;
    """
    model.eval()
    data_set_backwards = torch.flip(data_set, [1])
    missing_mask_backwards = torch.flip(missing_mask, [1])
    deltas_backwards = make_delta_simplified(missing_mask_backwards)
    with torch.no_grad():
        input = {
                    'forward': {'X': data_set * missing_mask,
                                'missing_mask': missing_mask,
                                'deltas': deltas},
                    'backward': {'X': data_set_backwards*missing_mask_backwards,
                                'missing_mask': missing_mask_backwards,
                                'deltas': deltas_backwards}
            }
        imputed_data, _ = model.impute(input)
        imputation_rmse = masked_rmse_cal(imputed_data, data_set, indicating_mask)
        imputation_mae = masked_mae_cal(imputed_data, data_set, indicating_mask)
        imputation_mre = masked_mre_cal(imputed_data, data_set, indicating_mask)

        return {
            'imputation rmse': imputation_rmse,
            'imputation mae': imputation_mae,
            'imputation mre': imputation_mre
        }

def mrnn_unwrap(config):
    """
    Parameters
    ----------
    config: dictionary containing:
        'seq_len':
              number of timesteps of the time series this model should impute
        'feat_num':
              number features of the timeseries this model should impute
        'rnn_hidden_size':
              hidden dimension of the RNN
        'device':
              which device should be used
    Returns
    -------
        M-RNN model:
            a new M-RNN model using the parameters meantioned above
    """
    return MRNN(config['seq_len'], config['feat_num'], config['rnn_hidden_size'], device=config['device'])\
        .to(config['device'])


def mrnn_train(config, enable_tracking = False, early_stopping = True):
    """
    Parameters
    ----------
    config: dictionary containing:
        'seq_len':
              number of timesteps of the time series this model should impute
        'feat_num':
              number features of the timeseries this model should impute
        'rnn_hidden_size':
              hidden dimension of the RNN
        'device':
              which device should be used
        'optimizer_name':
              name of the optimizer, that shall be usedin training
        'lr':
              learning rate, that shall be used in training
        'weigth_decay':
              weigth decay that shall be used in trainig
        'batch_size':
              batch size, that shall be used in training
        'patience':
              patience used for early stopping
        'epochs':
              maximal number of epochs of this training
        'train_set':
              training set
        'missing_mask_train':
              missing mask of the training set, respecting indicating mask
        'indicating_mask_train':
              indicating mask of the training set and missing mask
        'deltas_train':
            the delta tensor, containing the timesteps between the last seen values of the train_set
        'validation_set':
             validation set
        'missing_mask_validation':
              missing mask of the validation set, respecting indicating mask
        'indicating_mask_validation':
              indicating mask of the validation set and missing mask
        'deltas_validation':
            the delta tensor, containing the timesteps between the last seen values of the validation_set
        'enable_tracking':
              whether or not to track and output the validation metrics MAE, RMSE and MRE for every epoch
        'early_stopping':
              whether or not to use early stopping in this training
    Returns
    -------
        model:
            a new M-RNN models, trained by the parameters above
        tracker: dictionary
            contains the metrics MAE, RMSE and MRE of the validation set and the models estimation for every epoch of training
    """
    model = mrnn_unwrap(config)
    optimizer_name = config['optimizer_name']
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config['lr'],
                                                          weight_decay=config['weight_decay'])
    train_loader = DataLoader(range(config['train_set'].shape[0]), batch_size=config['batch_size'], shuffle=True)
    patience = 10  # fill this in as needed
    min_improvement = 0.001
    early_stopper = Early_stopper(patience=patience, min_improvement=min_improvement)
    if enable_tracking:
        tracker = {}
    for epoch in range(config['epochs']):
        model.train()
        mrnn_training_loop(train_loader, model, optimizer, config['train_set'], config['missing_mask_train'],
                           config['indicating_mask_train'], config['deltas_train'], config['device'])
        model.eval()
        with torch.no_grad():
            val_evaluation = mrnn_evaluate(model, config['validation_set'], config['missing_mask_validation'],
                                           config['indicating_mask_validation'], config['deltas_validation'])
        early_stopper.report(val_evaluation['imputation mae'].item())
        if enable_tracking:
            tracker[epoch] = val_evaluation
        if early_stopper.should_stop() & early_stopping:
            print('Early stopping at epoch: ', epoch)
            break
    if enable_tracking:
        return model, tracker
    return model

def mrnn_transfer_learning(config, enable_tracking = False, early_stopping = True):
    """
    Parameters
    ----------
    config: dictionary containing:
        'path_pretrained':
            the datapath to a pretrained M-RNN model, that shall be finetuned in this function
        'device':
              which device should be used
        'optimizer_name':
              name of the optimizer, that shall be usedin training
        'lr':
              learning rate, that shall be used in training
        'weigth_decay':
              weigth decay that shall be used in trainig
        'batch_size':
              batch size, that shall be used in training
        'patience':
              patience used for early stopping
        'epochs':
              maximal number of epochs of this training
        'train_set':
              training set
        'missing_mask_train':
              missing mask of the training set, respecting indicating mask
        'indicating_mask_train':
              indicating mask of the training set and missing mask
        'deltas_train':
            the delta tensor, containing the timesteps between the last seen values of the train_set
        'validation_set':
             validation set
        'missing_mask_validation':
              missing mask of the validation set, respecting indicating mask
        'indicating_mask_validation':
              indicating mask of the validation set and missing mask
        'deltas_validation':
              the delta tensor, containing the timesteps between the last seen values of the validation_set
        'enable_tracking':
              whether or not to track and output the validation metrics MAE, RMSE and MRE for every epoch
        'early_stopping':
              whether or not to use early stopping in this training
    Returns
    -------
        model:
            an M-RNN model, finetuned by the parameters above
        tracker: dictionary
            contains the metrics MAE, RMSE and MRE of the validation set and the models estimation for every epoch of training
    """
    model = torch.load(config['path_pretrained'])['model']
    train_set = config['train_set']
    missing_mask_train = config['missing_mask_train']
    indicating_mask_train = config['indicating_mask_train']
    validation_set = config['validation_set']
    missing_mask_validation = config['missing_mask_validation']
    indicating_mask_validation = config['indicating_mask_validation']
    deltas_train = config['deltas_train']
    deltas_validation = config['deltas_validation']
    device = config['device']
    optimizer_name = config['optimizer_name']
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr= config['lr'], weight_decay = config['weight_decay'])
    train_loader = DataLoader(range(train_set.shape[0]), batch_size= config['batch_size'], shuffle=True)
    patience = config['patience']
    min_improvement = 0.001
    early_stopper = Early_stopper(patience=patience, min_improvement=min_improvement)
    if enable_tracking:
        tracker = {}
    for epoch in range(config['epochs']):
        model.train()
        mrnn_training_loop(train_loader, model, optimizer, train_set, missing_mask_train,
                           indicating_mask_train, deltas_train, device)
        model.eval()
        with torch.no_grad():
            val_evaluation = mrnn_evaluate(model, validation_set, missing_mask_validation, indicating_mask_validation, deltas_validation)
        early_stopper.report(val_evaluation['imputation mae'].item())
        if enable_tracking:
            tracker[epoch] = val_evaluation
        if early_stopper.should_stop() & early_stopping:
            print('Early stopping at epoch: ', epoch)
            break
    if enable_tracking:
        return model, tracker
    return model

