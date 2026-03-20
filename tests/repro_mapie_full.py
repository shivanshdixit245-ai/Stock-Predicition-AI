import numpy as np
import pandas as pd
from mapie.classification import MapieClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional

def ensemble_predict_proba(ensemble, X):
    # Mocking ensemble_predict_proba
    return np.ones(len(X)) * 0.5

class EnsembleWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, ensemble: Optional[dict] = None):
        self.ensemble = ensemble
        
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        p1 = ensemble_predict_proba(self.ensemble, X)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
        
    def get_params(self, deep=True):
        return {"ensemble": self.ensemble}

def test():
    X_cal = pd.DataFrame(np.random.rand(10, 5))
    y_cal = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    ensemble = {}
    
    wrapper = EnsembleWrapper(ensemble)
    wrapper.fit(X_cal)
    
    try:
        mapie = MapieClassifier(estimator=wrapper, method='score', cv='prefit')
        mapie.fit(X_cal, y_cal)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
