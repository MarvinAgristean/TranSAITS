from torch.utils.data import DataLoader
from Models.SAITS.saits_utils import saits_unwrap, saits_evaluate, saits_training_loop
from Utils.early_stopper import Early_stopper
import torch
import optuna


# setting parameters
device = 'cpu'
if torch.cuda.is_available():
    device = "cuda:0"   # change to another gpu if needed
# data parameters
path_ALL_ICU = 'all_feat_all_icu_compatible'
path_HEART_ONLY = 'all_feat_heart_only_compatible'
path_pretrained_ALL_ICU = 'saits_all_feat_all_icu_best'
d_time = 48
d_feature = 10
# experiment parameters
fit_ALL_ICU = True
n_ALL_ICU_trials = 1
fit_HEART_ONLY = True
n_HEART_ONLY_trials = 1
fit_ALL_FIRST_HEART_SECOND = True
n_ALL_FIRST_HEART_SECOND_trials = 1
storage = 'sqlite:///saits.db'


def fit_data():
    # running the experiment according to the parameters above
    if fit_ALL_ICU:
        print('Start fitting ALL_ICU data...')
        study_name = 'SAITS_ALL_ICU'
        study = optuna.create_study(study_name=study_name, storage=storage, direction='minimize',
                                    load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=10, n_warmup_steps=10)
                                    )
        study.optimize(lambda trial: objective_one_dataset(trial, path_ALL_ICU), n_trials=n_ALL_ICU_trials, gc_after_trial=True)
        print('ALL_ICU best MAE on validation set: ', study.best_trial.value, 'with parameters: ', study.best_params)
        print('Finished fitting ALL_ICU data.')
    if fit_HEART_ONLY:
        print('Start fitting HEART_ONLY data...')
        study_name = 'SAITS_HEART_ONLY'
        study = optuna.create_study(study_name=study_name, storage=storage, direction='minimize',
                                    load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=10, n_warmup_steps=10)
                                    )
        study.optimize(lambda trial: objective_one_dataset(trial, path_HEART_ONLY), n_trials=n_HEART_ONLY_trials, gc_after_trial=True)
        print('HEART_ONLY best MAE on validation set: ', study.best_trial.value, 'with parameters: ', study.best_params)
        print('Finished fitting HEART_ONLY data.')
    if fit_ALL_FIRST_HEART_SECOND:
        print('Start fitting ALL_ICU first and HEART_ONLY second...')
        study_name = 'SAITS_ALL_FIRST_HEART_SECOND'
        study = optuna.create_study(study_name=study_name, storage=storage, direction='minimize',
                                    load_if_exists=True,
                                    pruner=optuna.pruners.MedianPruner(
                                        n_startup_trials=10, n_warmup_steps=10)
                                    )
        study.optimize(lambda trial: objective_two_datasets(trial, path_HEART_ONLY), n_trials=n_ALL_FIRST_HEART_SECOND_trials, gc_after_trial=True)
        print('ALL_ICU_FIRST_HEART_SECOND best MAE on validation set: ', study.best_trial.value, 'with parameters: ', study.best_params)
        print('Finished fitting ALL_ICU first and HEART_ONLY second.')

def objective_one_dataset(trial, path):
    config = {
            'device': device,
            'd_time': d_time,
            'd_feature': d_feature,
            'd_model': 2**trial.suggest_int('d_model_exponent', 4,7),
            'd_inner': 2**trial.suggest_int('d_inner_exponent', 4,7),
            'd_k': 2**trial.suggest_int('d_k_exponent', 4, 7),
            'd_v': 2**trial.suggest_int('d_v_exponent', 4,7),
            'n_head': 2**trial.suggest_int('n_head_exponent', 4, 7),
            'dropout': trial.suggest_float('dropout', 0, 1),
            'lr': trial.suggest_loguniform('lr', 0.0001, 1),
            'patience': 10,
            'weight_decay': trial.suggest_loguniform('weight_decay', 0.0001, 1),
            'momentum': 0,
            'epochs': 300,
            'batch_size': 2**trial.suggest_int('batch_size_exponent', 5, 7),
            'optimizer_name': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
        }
    print(config)
    # load data
    data_dic = torch.load(path)
    train_set = data_dic['train_set'].to(device)
    missing_mask_train = data_dic['missing_mask_train'].to(device)
    indicating_mask_train = data_dic['indicating_mask_train'].to(device)
    validation_set = data_dic['validation_set'].to(device)
    missing_mask_validation = data_dic['missing_mask_validation'].to(device)
    indicating_mask_validation = data_dic['indicating_mask_validation'].to(device)

    step_model = saits_unwrap(config)
    optimizer_name = config['optimizer_name']
    step_optimizer = getattr(torch.optim, optimizer_name)(step_model.parameters(), lr=config['lr'], weight_decay = config['weight_decay'])
    train_loader = DataLoader(range(train_set.shape[0]), batch_size=config['batch_size'], shuffle=True)
    patience = config['patience']
    min_improvement = 0.001
    early_stopper = Early_stopper(patience=patience, min_improvement=min_improvement)
    min_mae = torch.inf
    for epoch in range(config['epochs']):
        saits_training_loop(train_loader, step_model, step_optimizer, train_set, missing_mask_train,
                                   indicating_mask_train)
        val_evaluation = saits_evaluate(step_model, validation_set, missing_mask_validation, indicating_mask_validation)
        trial.report(val_evaluation['imputation mae'], epoch)
        # checking if the trial should end
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned
        # checking if the training is finished earlier
        early_stopper.report(val_evaluation['imputation mae'].item())
        if early_stopper.should_stop():
            print('Early stopping at epoch ', epoch)
            break
        if min_mae >= val_evaluation['imputation mae']:
            min_mae = val_evaluation['imputation mae']
    return min_mae.item()


def objective_two_datasets(trial, path):
    config = {
        'device': device,
        'd_time': d_time,
        'd_feature': d_feature,
        'lr': trial.suggest_loguniform('lr', 0.0001, 1),
        'patience': trial.suggest_int('patience', 0, 10),
        'weight_decay': trial.suggest_loguniform('weight_decay', 0.0001, 1),
        'epochs': 300,
        'batch_size': 2 ** trial.suggest_int('batch_size_exponent', 5, 7),
        'optimizer_name': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
    }
    print(config)
    # load data
    data_dic = torch.load(path)
    train_set = data_dic['train_set'].to(device)
    missing_mask_train = data_dic['missing_mask_train'].to(device)
    indicating_mask_train = data_dic['indicating_mask_train'].to(device)
    validation_set = data_dic['validation_set'].to(device)
    missing_mask_validation = data_dic['missing_mask_validation'].to(device)
    indicating_mask_validation = data_dic['indicating_mask_validation'].to(device)

    step_model = torch.load(path_pretrained_ALL_ICU)['model']
    optimizer_name = config['optimizer_name']
    step_optimizer = getattr(torch.optim, optimizer_name)(step_model.parameters(), lr=config['lr'])
    train_loader = DataLoader(range(train_set.shape[0]), batch_size=config['batch_size'], shuffle=True)
    patience = config['patience']
    min_improvement = 0.001
    early_stopper = Early_stopper(patience=patience, min_improvement=min_improvement)
    min_mae = torch.inf
    for epoch in range(config['epochs']):
        saits_training_loop(train_loader, step_model, step_optimizer, train_set, missing_mask_train, indicating_mask_train)
        val_evaluation = saits_evaluate(step_model, validation_set, missing_mask_validation, indicating_mask_validation)
        trial.report(val_evaluation['imputation mae'], epoch)
        # checking if the trial should end
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned
        # checking if the training is finished earlier
        early_stopper.report(val_evaluation['imputation mae'].item())
        if early_stopper.should_stop():
            print('Early stopping at epoch ', epoch)
            break
        if min_mae >= val_evaluation['imputation mae']:
            min_mae = val_evaluation['imputation mae']
    return min_mae.item()


if __name__ == "__main__":
    print('Start fitting data...')
    fit_data()
    print('Finished')



