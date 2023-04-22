import numpy as np
from sklearn.base import TransformerMixin


class ModelWrapper(TransformerMixin):
    """A convenient wrapper for final machine learning model
    which drops instances where target is unknown
    """
    def __init__(self, model, target_name, params={}):
        self.model = model(**params)
        self.target_name = target_name

    def fit(self, X, y):
        Xy = X.merge(y[['visit_id', self.target_name]], on='visit_id')
        Xy = Xy.dropna(subset=self.target_name)
        X_ordered = Xy.drop(['visit_id', self.target_name], axis=1)
        y_ordered = Xy[self.target_name]
        self.model.fit(X_ordered, y_ordered)
        return self

    def predict(self, X):
        return self.model.predict(X.drop('visit_id', axis=1))


def smape_func(y, y_pred, **kwargs):
    return 100/len(y) * np.sum(2 * np.abs(y_pred - y) /
                               (np.abs(y) + np.abs(y_pred)))
