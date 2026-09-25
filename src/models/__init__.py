from models.gru import GRUForecaster


# Teammates add their model classes here with the same (n_features, horizon, **kwargs) signature.
MODELS = {
    "gru": GRUForecaster,
}


def build_model(model_cfg, n_features, horizon):
    params = {key: value for key, value in model_cfg.items() if key != "name"}
    return MODELS[model_cfg["name"]](n_features=n_features, horizon=horizon, **params)
