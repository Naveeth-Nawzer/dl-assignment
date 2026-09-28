import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        max_len=500
    ):
        super().__init__()

        position = torch.arange(
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe = torch.zeros(
            max_len,
            d_model
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        sequence_length = x.size(1)

        return (
            x +
            self.pe[
                :, :sequence_length, :
            ]
        )


class TransformerRegressor(nn.Module):

    def __init__(
        self,
        input_features=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.2
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_features,
            d_model
        )

        self.positional_encoding = (
            PositionalEncoding(
                d_model=d_model,
                max_len=500
            )
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                d_model,
                32
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.positional_encoding(x)

        x = self.encoder(x)

        x = x[:, -1, :]

        x = self.regressor(x)

        return x.squeeze(-1)
