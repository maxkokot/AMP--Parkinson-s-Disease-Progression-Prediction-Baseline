import pandas as pd
import numpy as np
import re

from gensim.models import word2vec
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from tsfresh.feature_extraction import extract_features

SETTINGS_LEVEL_1 = {'sum_values': None,
                    'mean': None,
                    'maximum': None,
                    'minimum': None,
                    'standard_deviation': None,
                    'absolute_maximum':  None,
                    'autocorrelation': [{'lag': 0}, {'lag': 1},
                                        {'lag': 2}, {'lag': 3},
                                        {'lag': 4}, {'lag': 5},
                                        {'lag': 6}],
                    'agg_autocorrelation': [{'f_agg': 'mean', 'maxlag': 40},
                                            {'f_agg': 'median', 'maxlag': 40},
                                            {'f_agg': 'var', 'maxlag': 40}]
                    }

SETTINGS_LEVEL_2 = {'sum_values': {'mean': None},
                    'mean': {'mean': None},
                    'length': {'mean': None,
                               'maximum': None,
                               'minimum': None},
                    'maximum': {'maximum': None,
                                'mean': None},
                    'minimum': {'minimum': None,
                                'mean': None},
                    'standard_deviation': {'mean': None},
                    'autocorrelation': {'mean': None}}

SETTINGS_LEVEL_3 = {'sum_values': None,
                    'mean': None,
                    'maximum': None,
                    'minimum': None,
                    'standard_deviation': None,
                    'absolute_maximum':  None}


def create_config_pep(columns):

    config_pep = {col: {} for col in columns}
    config_pep[columns[0]].update({'length': None})

    for col in columns:
        config_pep[col].update(SETTINGS_LEVEL_1)

    return config_pep


def create_config_pep_prot(columns):

    config_pep_prot = {col: {} for col in columns}
    config_pep_prot[columns[0]].update({'length': None})

    for col in columns:
        for key in SETTINGS_LEVEL_2:
            if key in col:
                config_pep_prot[col] \
                    .update(SETTINGS_LEVEL_2[key])

    return config_pep_prot


def create_config_prot(columns):

    config_prot = {col: {} for col in columns}
    config_prot[columns[0]].update({'length': None})

    for col in columns:
        stop_indicator = False
        for key in SETTINGS_LEVEL_2:
            # stats for aggregative features
            if key in col:
                config_prot[col].update(SETTINGS_LEVEL_2[key])
                stop_indicator = True
        # stats for non aggregative features
        if not stop_indicator:
            config_prot[col].update(SETTINGS_LEVEL_3)

    return config_prot


class W2vPepWrapper(TransformerMixin):
    """A class for applying word2vec for
    encoding each character of peptides
    """
    def __init__(self, vector_size=30, window=5):
        self.vector_size = vector_size
        self.window = window
        self.cols = ['vector_pep_value_{}'.format(i)
                     for i in range(vector_size)]

    def _split_sequence(self, sequence):

        # We will separate single letters and UniMod strings
        # For example we want to split string 'ABCD(Unimod_4)EF like that:
        # A, B, C, D, Unimod, 4, E, F
        list_letters = sequence.split(r'(UniMod_')
        list_unimods = re.findall(r'\(UniMod_', sequence)

        list_letters = [letters.replace(')', '') for letters in list_letters]
        list_letters = [list(letters) for letters in list_letters]

        list_unimods = [unimod.replace(r'(', '') for unimod in list_unimods]
        list_unimods = [unimod.replace('_', '') for unimod in list_unimods]

        splitted_sequence = list_letters[0]

        if len(list_letters) > 1:

            for num in range(len(list_letters) - 1):
                splitted_sequence.append(list_unimods[num])
                splitted_sequence += list_letters[num + 1]

        elif len(list_unimods):
            splitted_sequence += list_unimods[0]

        return splitted_sequence

    def _create_vectors_df(self, splitted_sequence):
        vector_values = [self.w2v_model.wv.get_vector(symb)
                         for symb in splitted_sequence]
        vector_values = np.array(vector_values)
        vectors_df = pd.DataFrame(vector_values,
                                  columns=self.cols)
        vectors_df.index.name = 'Position'
        return vectors_df

    def _transform_peptides_to_vecs(self, peps):
        splitted_peps = self._split_sequence(peps.values[0])
        vec_df = self._create_vectors_df(splitted_peps)
        return vec_df

    def fit(self, X, y):
        peps, _, _ = X
        unique_peps = peps.Peptide.unique().astype('str')
        splitted_peps = [self._split_sequence(peptide_name)
                         for peptide_name in unique_peps]
        self.w2v_model = word2vec.Word2Vec(splitted_peps,
                                           vector_size=self.vector_size,
                                           window=self.window,
                                           workers=4)
        return self

    def transform(self, X):

        peps, prots, order = X
        peps_vectorized = peps.groupby('Peptide').Peptide \
            .apply(lambda x: self._transform_peptides_to_vecs(x))
        peps_vectorized = peps_vectorized.reset_index()
        return (peps, prots, order, peps_vectorized)


class W2vProtWrapper(TransformerMixin):
    """A class for applying word2vec for
    encoding the names of proteins
    """
    def __init__(self, vector_size=30, window=100):
        self.vector_size = vector_size
        self.window = window
        self.cols = ['vector_prot_value_{}'.format(i)
                     for i in range(vector_size)]

    def _create_vectors_df(self, protein):
        if protein in self.w2v_model.wv:
            vector_values = self.w2v_model.wv.get_vector(protein)
        else:
            vector_values = self.w2v_model.wv.get_mean_vector(protein)
        vector_values = np.array(vector_values).reshape(1, -1)
        vectors_df = pd.DataFrame(vector_values, columns=self.cols)
        return vectors_df

    def _transform_protein_to_vecs(self, protein):

        vec_df = self._create_vectors_df(protein.values[0])
        return vec_df

    def fit(self, X, y):
        _, _, _, agg_prots = X
        prot_combinations = agg_prots.groupby('visit_id')['UniProt'] \
            .apply(lambda x: x.values).values
        prot_combinations = [list(prot_combination)
                             for prot_combination in prot_combinations]
        self.w2v_model = word2vec.Word2Vec(prot_combinations,
                                           vector_size=self.vector_size,
                                           window=self.window,
                                           workers=4)
        return self

    def transform(self, X):
        peps, prots, order, agg_prots = X
        agg_prots_vectorized = agg_prots.copy()
        prot_vectors = agg_prots_vectorized.groupby(['visit_id', 'UniProt']) \
            .UniProt.apply(lambda x: self._transform_protein_to_vecs(x))
        prot_vectors = prot_vectors.reset_index()
        agg_prots_vectorized = agg_prots_vectorized \
            .merge(prot_vectors, on=['visit_id', 'UniProt'])
        agg_prots_vectorized = agg_prots_vectorized.drop('UniProt', axis=1)
        return (peps, prots, order, agg_prots_vectorized)


class TSHandler(TransformerMixin):
    """A class for transforming set of vectors
    into 1D vector
    """
    def __init__(self, chosen_idx, column_id, column_sort=None,
                 default_fc_parameters=None, config_function=None):
        self.chosen_idx = chosen_idx
        self.column_id = column_id
        self.column_sort = column_sort
        self.default_fc_parameters = default_fc_parameters
        self.config_function = config_function

    def fit(self, X, y):
        if self.config_function:
            self.default_fc_parameters = \
                self.config_function(X[self.chosen_idx].columns)

        return self

    def transform(self, X):
        agg_data = extract_features(X[self.chosen_idx],
                                    column_id=self.column_id,
                                    column_sort=self.column_sort,
                                    kind_to_fc_parameters=self
                                    .default_fc_parameters)
        agg_data.index.name = self.column_id
        agg_data = agg_data.reset_index()

        # replacing '__' with '_' to avoid problems further
        agg_data.columns = [col.replace('__', '_')
                            for col in agg_data.columns]
        final_list = list(X)
        final_list[self.chosen_idx] = agg_data
        final_tuple = tuple(final_list)
        return final_tuple


class PepInProtHandler(TransformerMixin):
    """A class for transforming set of vectors
    (which correspond to set of peptides
    for a certain protein belonging to
    certain visit) into 1D vector
    """
    def __init__(self, default_fc_parameters={'mean': None},
                 config_function=None):
        self.default_fc_parameters = default_fc_parameters
        self.config_function = config_function

        # We don't want to compute stats for following features
        # since they remain constant for visit-prot combinations
        self.out_of_agg_cols = ['visit_id', 'visit_month',
                                'patient_id', 'UniProt',
                                'NPX', 'Peptide',
                                'PeptideAbundance']

    def _get_pep_prot_combos(self, prot_data, pep_data):
        pep_prot = prot_data \
            .merge(pep_data, on=['visit_id', 'UniProt',
                                 'visit_month', 'patient_id'])[['visit_id',
                                                                'visit_month',
                                                                'patient_id',
                                                                'UniProt',
                                                                'NPX',
                                                                'Peptide']]

        pep_prot_combos = pep_prot \
            .groupby(['visit_id', 'UniProt'])[['UniProt', 'Peptide']] \
            .apply(lambda x: ' ' + '.'.join(x['Peptide']))
        pep_prot_combos = pep_prot_combos.reset_index()
        pep_prot_combos.columns = ['visit_id', 'UniProt', 'pep_prot_combo']
        return pep_prot_combos

    def _remove_duplicate_from_prot_data(self, prot_data, pep_prot_combos):
        pep_prot_combos_rem_dup = pep_prot_combos \
            .drop_duplicates(subset='pep_prot_combo')
        prot_data_rem_dup = prot_data.merge(pep_prot_combos_rem_dup,
                                            on=['visit_id', 'UniProt'])
        return prot_data_rem_dup

    def fit(self, X, y):
        if self.config_function:
            self.default_fc_parameters = self.config_function(X[-1].columns)

        return self

    def transform(self, X):

        peps, prots, order, agg_peps = X

        # Here we remove duplicated combinations of prots and peps
        # for decreasing computational time
        pep_prot_combos = self._get_pep_prot_combos(prots, peps)
        prots_rem_dup = self._remove_duplicate_from_prot_data(prots,
                                                              pep_prot_combos)

        pep_prot_rem_dup = prots_rem_dup \
            .merge(peps, on=['visit_id', 'UniProt',
                             'visit_month', 'patient_id'])
        pep_prot_rem_dup = pep_prot_rem_dup \
            .merge(agg_peps, on=['Peptide']) \
            .drop(self.out_of_agg_cols, axis=1)

        # feature extraction for peptides sets
        agg_agg_peps = extract_features(pep_prot_rem_dup,
                                        column_id='pep_prot_combo',
                                        kind_to_fc_parameters=self
                                        .default_fc_parameters)
        agg_agg_peps.index.name = 'pep_prot_combo'
        agg_agg_peps = agg_agg_peps.reset_index()

        # replacing '__' with '_' to avoid problems further
        agg_agg_peps.columns = [col.replace('__', '_')
                                for col in agg_agg_peps.columns]

        # feature extraction for PeptideAbundance
        agg_abundance = peps.groupby(['visit_id', 'UniProt']) \
            .apply(lambda x: x.PeptideAbundance.mean())
        agg_abundance = agg_abundance.reset_index()
        agg_abundance.columns = ['visit_id', 'UniProt', 'PeptideAbundance']

        # merge everything
        agg_pep_prots = prots \
            .merge(pep_prot_combos, on=['visit_id', 'UniProt']) \
            .merge(agg_agg_peps, on='pep_prot_combo')
        agg_pep_prots = agg_pep_prots.merge(agg_abundance,
                                            on=['visit_id', 'UniProt'])
        agg_pep_prots = agg_pep_prots.drop('pep_prot_combo', axis=1)
        return (peps, prots, order, agg_pep_prots)


class OrderMaintainer(TransformerMixin):
    """A class for maintaining
    initial order of visits
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        peps, prots, order, agg_df = X
        agg_df_ordered = order.merge(agg_df,
                                     on=list(order.columns))
        return (peps, prots, order, agg_df_ordered)


class PatDateExtractor(TransformerMixin):
    """A class for extracting patient_id
    and visit_month from visit_id
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def transform(self, X):
        peps, prots, order, agg_df = X
        agg_df_extracted = agg_df.copy()
        agg_df_extracted['patient_id'] = agg_df_extracted['visit_id'] \
            .apply(lambda x: x.split('_')[0]).astype('int')
        agg_df_extracted['visit_month'] = agg_df_extracted['visit_id'] \
            .apply(lambda x: x.split('_')[1]).astype('int')
        return (peps, prots, order, agg_df_extracted)


class ColumnDropper(TransformerMixin):
    """A class for dropping useless
    or leak features
    """
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X.drop(self.cols, axis=1)


class DFSelector(TransformerMixin):
    """A class for selecting a certain
    dataframe from a tuple of
    several dataframes
    """
    def __init__(self, chosen_idx):
        self.chosen_idx = chosen_idx

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X[self.chosen_idx]


class StandardScalerWrapper(StandardScaler):
    """A wrapper for StandardScaler
    returning pd.DataFrame
    """
    def __init__(self, *, copy=True,
                 with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean,
                         with_std=with_std)

    def transform(self, X):
        X_scaled = X.copy()
        X_scaled[X_scaled.columns] = super().transform(X)
        X_scaled['visit_id'] = X.visit_id
        return X_scaled


class FeatureSelectorWrapper(TransformerMixin):
    """A convenient wrapper for SelectFromModel
    """
    def __init__(self, model, target_name,
                 params={}, max_rate_feats=.5):
        self.params = params
        self.model = model
        self.target_name = target_name
        self.max_rate_feats = max_rate_feats

    def _compute_max_features(self, X):
        max_features = int(np.round(X.shape[1] * self.max_rate_feats))

        if max_features == 0:
            max_features = 1

        return max_features

    def fit(self, X, y):
        Xy = X.merge(y[['visit_id', self.target_name]], on='visit_id')
        Xy = Xy.dropna(subset=self.target_name)
        X_ordered = Xy.drop([self.target_name], axis=1)
        y_ordered = Xy[self.target_name]
        max_features = self._compute_max_features(X_ordered)
        self.selector = SelectFromModel(self.model(**self.params),
                                        threshold=-np.inf,
                                        max_features=max_features)
        self.selector.fit(X_ordered, y_ordered)
        return self

    def transform(self, X):
        arr = self.selector.transform(X).astype('float')
        cols = self.selector.get_feature_names_out()
        df = pd.DataFrame(arr, columns=cols)
        df['visit_id'] = X['visit_id']
        return df
