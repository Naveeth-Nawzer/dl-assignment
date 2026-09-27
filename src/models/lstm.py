
from torch import nn


class LSTMForecaster(nn.Module):
    """
    Stacked LSTM encoder with a linear head for direct multi-step forecasts.

    Input:
        [batch, lookback, n_features]

    Output:
        [batch, horizon]
    """

    def __init__(
        self,
        n_features,
        horizon,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.head = nn.Linear(
            hidden_size,
            horizon
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)

        return self.head(
            self.dropout(hidden[-1])
        )
