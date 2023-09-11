import torch
from torch.utils.data import DataLoader
from Models.SAITS.saits import SAITS
from Utils.metric_service import masked_mae_cal, masked_rmse_cal, masked_mre_cal
from Utils.early_stopper import Early_stopper
import gc  # garbage collector


def saits_training_loop(loader, model, optimizer, data_set, missing_mask, indicating_mask ):
    """
    Parameters
    ----------
    loader: DataLoader
        dataloader of the indice of the data_set, missing_mask and indicating_mask
    model: SAITS
        SAITS model that shall be trained here
    optimizer:
        the optimizer that shall be used in this training epoch
    data_set: torch tensor
        dataset used for training
    missing_mask: torch tensor
        missing mask of the data_set, already contains artifially masked data
    indicating_mask: torch tensor
        indicating mask of the data_set and missing_mask
    Returns
    -------
    number:
        the mean loss of this epoch
    """
    model.train()
    mean_loss = 0
    counter = 0+1e-9
    for indice in loader:
        counter = counter + 1
        data_batch = data_set[indice]
        missing_mask_batch = missing_mask[indice]
        indicating_mask_batch = indicating_mask[indice]
        input = {
            'X': (data_batch * (torch.ones_like(data_batch) - indicating_mask_batch)),  # the model shouldn't see indicated values
            'missing_mask' : missing_mask_batch,
            'indicating_mask' : indicating_mask_batch,
            'X_holdout' : data_batch,
            }
        output = model.forward(input, 'test')  # test stage?
        loss = output['reconstruction_loss'] + output['imputation_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_loss += loss
    return mean_loss/counter


def saits_evaluate(model, data_set, missing_mask, indicating_mask, batch_size=33):
    """
    Parameters
    ----------
    model: SAITS
        SAITS model that shall be trained here
    data_set: torch tensor
        dataset used for evalutation, must also contain the holdout values
    missing_mask: torch tensor
        missing mask of the data_set, already contains artifially masked data
    indicating_mask: torch tensor
        indicating mask of the data_set and missing_mask
    batch_size:
        the batch sized used in the evaluation
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
    # data_set must also contain the indicated values; change batch size if torch.cat throws error
    model.eval()
    data_loader = DataLoader(range(data_set.shape[0]), batch_size=batch_size, shuffle=False)
    imputed_data = torch.zeros_like(data_set)
    for batch in data_loader:
        data_batch = data_set[batch]
        missing_mask_batch = missing_mask[batch]
        input_batch = {
            'X': (data_batch * missing_mask_batch),  # the model shouldn't see missing/indicated values
            'missing_mask': missing_mask_batch
            }
        with torch.no_grad():
            imputed_data_batch, _ = model.impute(input_batch)
        imputed_data[batch] = imputed_data_batch
    imputation_rmse = masked_rmse_cal(imputed_data, data_set,indicating_mask)
    imputation_mae = masked_mae_cal(imputed_data, data_set,indicating_mask)
    imputation_mre = masked_mre_cal(imputed_data, data_set,indicating_mask)
    return {
            'imputation rmse': imputation_rmse,
            'imputation mae': imputation_mae,
            'imputation mre': imputation_mre
        }


def saits_unwrap(config):
    """
    Parameters
    ----------
    config: dictionary containing:
        'd_time':
              number of timesteps of the time series this model should impute
        'd_feature':
              number features of the timeseries this model should impute
        'd_model':
              hidden dimension frequently used to represent data processed between the steps
        'd_inner':
              hidden dimension of feed forward layers
        'd_k':
              dimension of key in attention
        'd_v':
              dimension of value in attention
        'dropout':
              dropoutrate
        'n_groups':
              number of groups of inner layers
        'n_group_inner_layers':
              number of inner layers in a group
        'n_head':
              number of attentionheads
        'device':
              which device should be used
    Returns
    -------
        SAITS model:
            a new SAITS models using the parameters meantioned above
    """
    return SAITS(config['d_time'], config['d_feature'], config['d_model'],
                 config['d_inner'], config['d_k'], config['d_v'],
                 n_head=config['n_head'], device=config['device'], dropout=config['dropout']).to(config['device'])

def saits_train(config, enable_tracking = False, early_stopping = True):
    """
    Parameters
    ----------
    config: dictionary containing:
        'd_time':
              number of timesteps of the time series this model should impute
        'd_feature':
              number features of the timeseries this model should impute
        'd_model':
              hidden dimension frequently used to represent data processed between the steps
        'd_inner':
              hidden dimension of feed forward layers
        'd_k':
              dimension of key in attention
        'd_v':
              dimension of value in attention
        'dropout':
              dropoutrate
        'n_groups':
              number of groups of inner layers
        'n_group_inner_layers':
              number of inner layers in a group
        'n_head':
              number of attentionheads
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
        'validation_set':
             validation set
        'missing_mask_validation':
              missing mask of the validation set, respecting indicating mask
        'indicating_mask_validation':
              indicating mask of the validation set and missing mask
        'enable_tracking':
              whether or not to track and output the validation metrics MAE, RMSE and MRE for every epoch
        'early_stopping':
              whether or not to use early stopping in this training
    Returns
    -------
        model:
            a new SAITS models, trained by the parameters above
        tracker: dictionary
            contains the metrics MAE, RMSE and MRE of the validation set and the models estimation for every epoch of training
    """
    # check this according to the hyperparamter optimization
    model = saits_unwrap(config)
    optimizer_name = config['optimizer_name']
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
    train_loader = DataLoader(range(config['train_set'].shape[0]), batch_size=config['batch_size'], shuffle=True)
    patience = config['patience']
    min_improvement = 0.001
    early_stopper = Early_stopper(patience=patience, min_improvement=min_improvement)
    if enable_tracking:
        tracker = {}
    for epoch in range(config['epochs']):
        saits_training_loop(train_loader, model, optimizer, config['train_set'], config['missing_mask_train'],
                                   config['indicating_mask_train'])
        val_evaluation = saits_evaluate(model, config['validation_set'], config['missing_mask_validation'], config['indicating_mask_validation'])
        # checking if the training is finished earlier
        early_stopper.report(val_evaluation['imputation mae'].item())
        if enable_tracking:
            tracker[epoch] = val_evaluation
        if early_stopper.should_stop() & early_stopping:
            print('Early stopping at epoch ', epoch)
            break
    if enable_tracking:
        return model, tracker
    return model

def saits_transfer_learning(config, enable_tracking = False, early_stopping = True):
    """
    Parameters
    ----------
    config: dictionary containing:
        'path_pretrained':
            the datapath to a pretrained SAITS model, that shall be finetuned in this function
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
        'validation_set':
             validation set
        'missing_mask_validation':
              missing mask of the validation set, respecting indicating mask
        'indicating_mask_validation':
              indicating mask of the validation set and missing mask
        'enable_tracking':
              whether or not to track and output the validation metrics MAE, RMSE and MRE for every epoch
        'early_stopping':
              whether or not to use early stopping in this training
    Returns
    -------
        model:
            a SAITS models, finetuned by the parameters above
        tracker: dictionary
            contains the metrics MAE, RMSE and MRE of the validation set and the models estimation for every epoch of training
    """
    model = torch.load(config['path_pretrained'])['model']
    optimizer_name = config['optimizer_name']
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=config['lr'])
    train_set = config['train_set']
    missing_mask_train = config['missing_mask_train']
    indicating_mask_train = config['indicating_mask_train']
    validation_set = config['validation_set']
    missing_mask_validation = config['missing_mask_validation']
    indicating_mask_validation = config['indicating_mask_validation']
    train_loader = DataLoader(range(train_set.shape[0]), batch_size=config['batch_size'], shuffle=True)
    patience = config['patience']
    min_improvement = 0.001
    early_stopper = Early_stopper(patience=patience, min_improvement=min_improvement)
    if enable_tracking:
        tracker = {}
    for epoch in range(config['epochs']):
        saits_training_loop(train_loader, model, optimizer, train_set, missing_mask_train, indicating_mask_train)
        with torch.no_grad():
            val_evaluation = saits_evaluate(model, validation_set, missing_mask_validation, indicating_mask_validation)
        # cleaning memory
        torch.cuda.empty_cache()
        gc.collect()
        # checking if the training is finished earlier
        early_stopper.report(val_evaluation['imputation mae'].item())
        if enable_tracking:
            tracker[epoch] = val_evaluation
        if early_stopper.should_stop() & early_stopping:
            print('Early stopping at epoch ', epoch)
            break
    if enable_tracking:
        return model, tracker
    return model

