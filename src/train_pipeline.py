import click
import numpy as np
import os
import logging
import sys

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.tree import ExtraTreeRegressor
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor
import optuna
import pickle

from data.make_dataset import read_data, split_train_val_data, \
    split_cv_data
from features.build_features import W2vPepWrapper, TSHandler, \
    PepInProtHandler, OrderMaintainer, W2vProtWrapper, \
    PatDateExtractor, DFSelector, ColumnDropper, \
    StandardScalerWrapper, FeatureSelectorWrapper, \
    create_config_pep, create_config_pep_prot, create_config_prot
from models.train_model import smape_func, ModelWrapper


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

current_target = None
train_peptides = None
train_proteins = None
train_clinical = None

MODELS = {'ExtraTreeRegressor': ExtraTreeRegressor,
          'Ridge': Ridge
          }

COL_TO_DEL = ['patient_id', 'patient_id_sum_values', 'patient_id_mean',
              'patient_id_maximum', 'patient_id_minimum',
              'patient_id_standard_deviation', 'patient_id_absolute_maximum',
              'visit_month_sum_values', 'visit_month_mean',
              'visit_month_maximum', 'visit_month_minimum',
              'visit_month_standard_deviation', 'visit_month_absolute_maximum']

TARGETS = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']


def make_pipeline(pep_vector_size, pep_window_size,
                  prot_vector_size, prot_window_size,
                  model_for_fs, rate_for_fs,
                  max_depth, learning_rate,
                  n_estimators, min_child_weight,
                  gamma, current_target):

    param_xgb = {
        'max_depth': max_depth,
        'learning_rate': learning_rate,
        'n_estimators': n_estimators,
        'min_child_weight': min_child_weight,
        'gamma': gamma}

    pipeline = Pipeline([('w2pepvw', W2vPepWrapper(pep_vector_size,
                                                   pep_window_size)),
                         ('tsh1', TSHandler
                          (3, 'Peptide', 'Position',
                           config_function=create_config_pep)),
                         ('ph', PepInProtHandler
                          (config_function=create_config_pep_prot)),
                         ('om1', OrderMaintainer()),
                         ('w2protvw', W2vProtWrapper(prot_vector_size,
                                                     prot_window_size)),
                         ('tsh2', TSHandler
                          (3, 'visit_id', None,
                           config_function=create_config_prot)),
                         ('pde', PatDateExtractor()),
                         ('om2', OrderMaintainer()),
                         ('dfs', DFSelector(-1)),
                         ('cd', ColumnDropper(COL_TO_DEL)),
                         ('sc', StandardScalerWrapper()),
                         ('fsel', FeatureSelectorWrapper
                          (MODELS[model_for_fs],
                           current_target,
                           max_rate_feats=rate_for_fs)),
                         ('model', ModelWrapper(XGBRegressor,
                                                current_target,
                                                params=param_xgb))
                         ])

    return pipeline


def objective(trial):

    global current_target, train_peptides, train_proteins, train_clinical

    pep_vector_size = trial.suggest_int("pep_vector_size", 5, 50)
    pep_window_size = trial.suggest_int("pep_window_size", 3, 20)

    prot_vector_size = trial.suggest_int("prot_vector_size", 5, 50)
    prot_window_size = trial.suggest_int("prot_window_size", 3, 10)

    model_for_fs = trial.suggest_categorical('model_for_fs',
                                             ['ExtraTreeRegressor', 'Ridge'])
    rate_for_fs = trial.suggest_float('rate_for_fs', 0.01, 0.99)

    max_depth = trial.suggest_int('max_depth', 1, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0)
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    gamma = trial.suggest_float('gamma', 0.01, 1.0)

    pipe = make_pipeline(pep_vector_size, pep_window_size,
                         prot_vector_size, prot_window_size,
                         model_for_fs, rate_for_fs,
                         max_depth, learning_rate,
                         n_estimators, min_child_weight,
                         gamma, current_target)

    cv = KFold(3, shuffle=True, random_state=22)
    patient_ids = train_peptides.patient_id.unique()
    smape_scores = []

    for train_idxs, val_idxs in cv.split(patient_ids):

        smape_score = _fit_and_score_cv(pipe, train_peptides,
                                        train_proteins,
                                        train_clinical,
                                        current_target,
                                        train_idxs, val_idxs)
        smape_scores.append(smape_score)

    mean_smape_score = np.mean(smape_scores)
    return mean_smape_score


def _fit_and_score_cv(pipe, peps, prots,
                      clinical, current_target,
                      train_idxs, val_idxs):
    patient_ids = peps.patient_id.unique()
    train_ids = patient_ids[train_idxs]
    val_ids = patient_ids[val_idxs]

    train_peps, val_peps, \
        train_prots, val_prots, \
        train_clinical, val_clinical = \
        split_cv_data(peps, prots,
                      clinical, train_ids, val_ids)

    train_order = train_peps[[
        'patient_id', 'visit_month']].drop_duplicates()
    val_order = val_peps[[
        'patient_id', 'visit_month']].drop_duplicates()

    pipe.fit((train_peps,
              train_prots,
              train_order), train_clinical)
    preds = pipe.predict((val_peps,
                          val_prots,
                          val_order))

    preds_val_df = val_order.copy()
    preds_val_df['pred'] = preds

    true_pred = preds_val_df \
        .merge(val_clinical, on=['patient_id',
                                 'visit_month'])
    true_pred = true_pred.dropna(subset=current_target)

    smape_score = smape_func(true_pred[current_target].values,
                             true_pred['pred'].values)
    return smape_score


def train(peptides_path,
          proteins_path,
          clinical_path,
          model_path,
          n_iter=10):

    global current_target, train_peptides, \
        train_proteins, train_clinical

    peptides = read_data(peptides_path)
    proteins = read_data(proteins_path)
    clinical = read_data(clinical_path)

    train_peptides, test_peptides, \
        train_proteins, test_proteins, \
        train_clinical, test_clinical = split_train_val_data(peptides,
                                                             proteins,
                                                             clinical,
                                                             test_size=.2,
                                                             random_state=20)

    logger.info(f"train_peptides.shape is {train_peptides.shape}")
    logger.info(f"train_proteins.shape is {train_proteins.shape}")
    logger.info(f"train_clinical.shape is {train_clinical.shape}")

    logger.info(f"test_peptides.shape is {test_peptides.shape}")
    logger.info(f"test_proteins.shape is {test_proteins.shape}")
    logger.info(f"test_clinical.shape is {test_clinical.shape}")

    train_order = train_proteins[[
        'patient_id', 'visit_month']].drop_duplicates()
    test_order = test_proteins[[
        'patient_id', 'visit_month']].drop_duplicates()

    logger.info("starting trainig")
    for current_target in TARGETS:
        logger.info(f"current target is {current_target}")
        current_study = optuna.create_study()
        current_study.optimize(objective, n_trials=n_iter)

        current_pipe = make_pipeline(current_target=current_target,
                                     **current_study.best_params)
        current_pipe.fit((train_peptides,
                          train_proteins,
                          train_order), train_clinical)
        preds_test = current_pipe.predict((test_peptides,
                                           test_proteins,
                                           test_order))
        preds_test_df = test_order.copy()
        preds_test_df['pred'] = preds_test
        true_pred = preds_test_df.merge(test_clinical, on=['patient_id',
                                                           'visit_month'])
        true_pred = true_pred.dropna(subset=current_target)
        current_smape = smape_func(true_pred.updrs_1.values,
                                   true_pred['pred'].values)
        logger.info(f"current smape is {current_smape}")
        with open(os.path.join(model_path,
                               '{}.pkl'.format(current_target)),
                  "wb") as f:
            pickle.dump(current_pipe, f)


@click.command(name="train_pipeline")
@click.option('--peptides_path', default='../data/raw/train_peptides.csv')
@click.option('--proteins_path', default='../data/raw/train_proteins.csv')
@click.option('--clinical_path', default='../data/raw/train_clinical_data.csv')
@click.option('--model_path', default='../models')
@click.option('--n_iter', default=10)
def train_pipeline_command(peptides_path,
                           proteins_path,
                           clinical_path,
                           model_path,
                           n_iter):
    train(peptides_path,
          proteins_path,
          clinical_path,
          model_path,
          n_iter)


if __name__ == "__main__":
    train_pipeline_command()
