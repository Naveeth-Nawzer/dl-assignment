import numpy as np
import torch


def window_index(first_valid, start, end, cfg, stride):
    """
    Return (series, t) pairs for windows whose targets fall in days [start, end).

    The input covers days [t - lookback, t) and the target covers [t, t + horizon).
    """
    lookback = cfg["lookback"]
    horizon = cfg["horizon"]

    pairs = []
    for series, first in enumerate(first_valid):
        # The input window starts after one lookback of history, so the 28-day rolling mean is complete.
        earliest = max(start, int(first) + 2 * lookback)
        targets = np.arange(earliest, end - horizon + 1, stride)
        pairs.append(np.stack([np.full_like(targets, series), targets], axis=1))

    return np.concatenate(pairs).astype(np.int64)


def split_indices(data, cfg):
    """
    Return the train, validation and test window indices.

    Validation and test targets never overlap with training targets.
    """
    first_valid = data["first_valid"]

    return {
        "train": window_index(
            first_valid, 0, cfg["train_end"], cfg, cfg["train_stride"]
        ),
        "val": window_index(
            first_valid, cfg["train_end"], cfg["val_end"], cfg, cfg["eval_stride"]
        ),
        "test": window_index(
            first_valid, cfg["val_end"], cfg["n_days"], cfg, cfg["eval_stride"]
        ),
    }


class WindowLoader:
    """
    Build batches of windows by indexing tensors that stay on the device.

    This is much faster than a per-sample Dataset for millions of short windows.
    """

    def __init__(self, data, index, cfg, batch_size, device,
                 shuffle=False, seed=0, max_batches=None):
        self.features = torch.as_tensor(data["features"], device=device)
        self.targets = torch.as_tensor(
            data["sales"] / data["scale"][:, None], device=device
        )
        self.index = torch.as_tensor(index, device=device)
        self.input_offsets = torch.arange(-cfg["lookback"], 0, device=device)
        self.target_offsets = torch.arange(cfg["horizon"], device=device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_batches = max_batches
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self):
        n_batches = -(-len(self.index) // self.batch_size)
        if self.max_batches is not None:
            n_batches = min(n_batches, self.max_batches)
        return n_batches

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.index), generator=self.generator)
            order = order.to(self.index.device)
        else:
            order = torch.arange(len(self.index), device=self.index.device)

        for batch in range(len(self)):
            rows = self.index[order[batch * self.batch_size:(batch + 1) * self.batch_size]]
            series = rows[:, 0:1]
            t = rows[:, 1:2]

            x = self.features[series, t + self.input_offsets]
            y = self.targets[series, t + self.target_offsets]

            yield x, y, rows
