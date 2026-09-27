
from models.gru import GRUForecaster
from models.lstm import LSTMForecaster


# Model registry
MODELS = {
    "gru": GRUForecaster,
    "lstm": LSTMForecaster,
}


def build_model(model_cfg, n_features, horizon):
    params = {
        key: value
        for key, value in model_cfg.items()
        if key != "name"
    }

    return MODELS[model_cfg["name"]](
        n_features=n_features,
        horizon=horizon,
        **params
    )
