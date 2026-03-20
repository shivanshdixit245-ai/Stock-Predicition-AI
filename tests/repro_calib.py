import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

import sklearn
def test():
    print(f"Sklearn version: {sklearn.__version__}")
    X = np.random.rand(20, 5)
    y = np.array([0, 1] * 10)
    
    model = LogisticRegression().fit(X, y)
    from sklearn.utils.validation import check_is_fitted
    try:
        check_is_fitted(model)
        print("Model is fitted.")
        cal = CalibratedClassifierCV(estimator=model, method='sigmoid', cv=None)
        cal.fit(X, y)
        print("SUCCESS")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
