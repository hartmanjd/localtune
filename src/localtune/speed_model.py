"""Writing-speed guesses from the trained prediction model, when one is on this computer."""

# The trained prediction model (scripts/train.py) is kept as one file in localtune's own folder
# (results.home_dir), never in the project, so it is only on the computers it was copied to.
# Where it isn't, or where XGBoost isn't installed (it's only in the train group, not in the
# normal install), nothing here does anything and 'localtune predict' shows the formula alone.
#
# The model learned how many times faster or slower than the formula (predict.py) real runs
# were, from the inputs in features.py. So a guess is the formula's estimate times that learned
# factor, for a row of run data shaped like the ones it was trained on. A model file saved with
# different inputs than this version of localtune works out is ignored, because its guesses
# would mean nothing.

import math
from pathlib import Path

from localtune.features import FEATURES, inputs, number
from localtune.results import home_dir


def model_file() -> Path:
    return home_dir() / "speed-model.json"


def load_model():
    path = model_file()
    if not path.exists():
        return None
    try:
        import xgboost
    except ImportError:
        return None
    model = xgboost.Booster()
    model.load_model(str(path))
    return model if model.feature_names == FEATURES else None


def trained_estimate(row: dict) -> float | None:
    estimate = number(row, "estimate.decode_tok_s")
    model = load_model()
    if model is None or math.isnan(estimate):
        return None
    import numpy as np
    import xgboost

    data = xgboost.DMatrix(np.array([inputs(row)]), feature_names=FEATURES)
    return estimate * math.exp(float(model.predict(data)[0]))
