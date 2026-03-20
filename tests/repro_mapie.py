import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import joblib

class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble=None):
        self.ensemble = ensemble
        self.classes_ = np.array([0, 1])
        self.fitted_ = True
    def predict_proba(self, X):
        # Dummy proba
        return np.column_stack([np.ones(len(X))*0.5, np.ones(len(X))*0.5])
    def predict(self, X):
        return np.zeros(len(X))
    def fit(self, X, y):
        return self

def test():
    X = pd.DataFrame(np.random.rand(10, 5))
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    wrapper = EnsembleWrapper({})
    try:
        mapie = MapieClassifier(estimator=wrapper, method='score', cv='prefit')
        mapie.fit(X, y)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
