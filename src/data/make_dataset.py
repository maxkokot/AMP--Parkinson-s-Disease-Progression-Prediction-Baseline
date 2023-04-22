# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split


def read_data(path: str):
    data = pd.read_csv(path)
    return data


def _split_all_data(peps, prots, clinical, train_patients, val_patients):
    train_peps = peps[peps.patient_id.isin(train_patients)] \
        .reset_index(drop=True)
    val_peps = peps[peps.patient_id.isin(val_patients)] \
        .reset_index(drop=True)

    train_prots = prots[prots.patient_id.isin(train_patients)] \
        .reset_index(drop=True)
    val_prots = prots[prots.patient_id.isin(val_patients)] \
        .reset_index(drop=True)

    train_clinical = clinical[clinical.patient_id.isin(train_patients)] \
        .reset_index(drop=True)
    val_clinical = clinical[clinical.patient_id.isin(val_patients)] \
        .reset_index(drop=True)
    return train_peps, val_peps, train_prots, val_prots, \
        train_clinical, val_clinical


def split_train_val_data(peps, prots, clinical, test_size, random_state):
    train_patients, val_patients = train_test_split(peps.patient_id.unique(),
                                                    test_size=test_size,
                                                    random_state=random_state)

    train_peps, val_peps, train_prots, val_prots, \
        train_clinical, val_clinical = _split_all_data(peps, prots,
                                                       clinical,
                                                       train_patients,
                                                       val_patients)

    return train_peps, val_peps, train_prots, val_prots, \
        train_clinical, val_clinical


def split_cv_data(peps, prots, clinical, train_idxs, val_idxs):

    train_peps, val_peps, train_prots, val_prots, \
        train_clinical, val_clinical = _split_all_data(peps, prots,
                                                       clinical,
                                                       train_idxs,
                                                       val_idxs)

    return train_peps, val_peps, train_prots, val_prots, \
        train_clinical, val_clinical
