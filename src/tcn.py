import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    Removes extra values introduced by padding so that
    the temporal output remains aligned with the input.
    """

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    One residual TCN block containing two causal dilated
    convolution layers.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.2
    ):
        super().__init__()

        padding = (
            kernel_size - 1
        ) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp1 = Chomp1d(
            padding
        )

        self.activation1 = nn.ReLU()

        self.dropout1 = nn.Dropout(
            dropout
        )

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.chomp2 = Chomp1d(
            padding
        )

        self.activation2 = nn.ReLU()

        self.dropout2 = nn.Dropout(
            dropout
        )

        if in_channels != out_channels:
            self.residual = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )
        else:
            self.residual = nn.Identity()

        self.final_activation = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.activation2(out)
        out = self.dropout2(out)

        residual = self.residual(x)

        return self.final_activation(
            out + residual
        )


class TCNRegressor(nn.Module):
    """
    Temporal Convolutional Network for next-day
    retail demand forecasting.
    """

    def __init__(
        self,
        input_features=1,
        channels=(32, 32, 64, 64),
        kernel_size=3,
        dropout=0.2
    ):
        super().__init__()

        layers = []

        for i, out_channels in enumerate(channels):

            in_channels = (
                input_features
                if i == 0
                else channels[i - 1]
            )

            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(
            *layers
        )

        self.regressor = nn.Sequential(
            nn.Linear(
                channels[-1],
                32
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                32,
                1
            )
        )

    def forward(self, x):

        # Input:
        # [batch, time, features]

        x = x.transpose(
            1,
            2
        )

        # [batch, features, time]

        x = self.tcn(x)

        # Take final temporal position
        x = x[:, :, -1]

        # Regression head
        x = self.regressor(x)

        return x.squeeze(-1)
