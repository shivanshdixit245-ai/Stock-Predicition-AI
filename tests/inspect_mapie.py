import joblib
from pathlib import Path
from mapie.classification import MapieClassifier

def test():
    ticker = "AAPL"
    model_path = Path("data/models") / ticker / "mapie_conformal.pkl"
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return
    
    mapie = joblib.load(model_path)
    print(f"Mapie object type: {type(mapie)}")
    print(f"Attributes: {dir(mapie)}")
    
    if hasattr(mapie, "estimator"):
        print(f"Has 'estimator': {type(mapie.estimator)}")
    if hasattr(mapie, "estimator_"):
        print(f"Has 'estimator_': {type(mapie.estimator_)}")
    else:
        print("Does NOT have 'estimator_'")

if __name__ == "__main__":
    test()
