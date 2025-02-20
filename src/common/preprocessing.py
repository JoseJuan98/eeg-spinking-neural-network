# -*- coding: utf-8 -*-
"""
These modules are based on code originally developed by quadrater/eeg.
The original code can be found at: https://github.com/quadrater/eeg

All credit for the original implementation goes to the contributors of the above repository.
This script may include modifications or extensions made for specific use cases.

Original Repository License: None
Please ensure compliance with the original license when using or distributing this code.

Modifications by: (I) josejuan98
"""
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import picard
import scipy.io
import scipy.signal
import torch
import tqdm
from sklearn.preprocessing import RobustScaler

from common.config import Config
from common.log import get_logger


def get_dataset(file_name: str | pathlib.Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load the dataset from the file.

    Args:
        file_name (str | pathlib.Path): The file name of the dataset inside the artifacts/data dir.

    Returns:
        torch.Tensor: The data tensor.
        torch.Tensor: The labels' tensor.
        torch.Tensor: The subjects' tensor.
        dict: metadata information.
    """
    dataset = torch.load(f=Config.data_dir / file_name, weights_only=True)

    return dataset["data"], dataset["labels"], dataset["subjects"], dataset["metadata"]


def normalize(data: torch.Tensor, dim: int = None) -> torch.Tensor:
    means = data.mean(dim=dim, keepdim=True)
    stds = data.std(dim=dim, keepdim=True)
    data -= means
    data /= stds
    return data


def common_average_referencing(data: torch.Tensor, dim: int = None, residual: bool = False) -> torch.Tensor:
    means = data.mean(dim=dim, keepdim=True)
    data -= means
    if residual:
        return torch.cat([data, means], dim=dim)
    return data


# Super naive spike thesholdning. Everything beyond +/- 1 SD is a spike.
def abs_threshold(data: torch.Tensor, threshold: float = 1) -> torch.Tensor:
    return (data.abs() > 1).float()


# Naive spike thesholdning. Everything beyond +/- 1 SD is a spike.
def pol_threshold(data: torch.Tensor, threshold: float = 1, dim: int = 1) -> torch.Tensor:
    return torch.cat(((data > threshold).float(), (data < -threshold).float()), dim=dim)


def delta_coding(data: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    quantized = (data / delta).to(torch.int)
    diff = torch.diff(quantized, dim=2, prepend=data[:, :, :1])

    # This wasn't exactly obvious. floor(1.5) is 1 while floor (-1.5) is -2
    spikes_delta = torch.cat(
        (
            torch.floor(diff).clamp(min=0),
            -torch.ceil(diff).clamp(max=0),
        ),
        dim=1,
    ).to(torch.float32)
    return spikes_delta


def rate_coding(data: torch.Tensor, sensitivity: float = 0.1) -> torch.Tensor:
    spikes_quant = data / sensitivity
    spikes_rate = (
        torch.cat(
            (
                torch.floor(spikes_quant).clamp(min=0),
                -torch.ceil(spikes_quant).clamp(max=0),
            ),
            dim=1,
        )
        .to(torch.int)
        .to(torch.float32)
    )
    return spikes_rate


def bin_sum(data: torch.Tensor, size: int, dim: int = -1):
    """Bin the data into fewer bins by summation, analogous to Tonic.ToFrame. Implementation by ChatGPT."""
    total_indices = data.size(dim)
    quotient, remainder = divmod(total_indices, size)
    bin_sizes = [quotient + (1 if i >= size - remainder else 0) for i in range(size)]
    splits = torch.split(torch.arange(total_indices), bin_sizes)
    return torch.stack([data.index_select(dim, idx).sum(dim=dim) for idx in splits], dim=dim)


def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    events, targets = zip(*batch)
    events = torch.stack(events).permute(2, 0, 1)
    targets = torch.stack(targets)
    return events, targets


def downsample(data: torch.Tensor, in_rate: int = 1024, out_rate: int = 1024 // 8):
    samples = data.size(-1) * out_rate // in_rate
    result = scipy.signal.resample(data.cpu().numpy(), samples, axis=-1)
    return torch.tensor(result, device=data.device, dtype=data.dtype)


def ica(
    data: torch.Tensor, max_iter: int = 500, tol: float = 1e-8, whiten: bool = True, ortho: bool = False
) -> torch.Tensor:
    """Independent Component Analysis (ICA) using the Picard library."""
    results = []
    print("ICA")
    for d in tqdm.tqdm(iterable=data, desc="ICA", unit="trials", leave=False):
        d = d.transpose(0, 1)
        transform = picard.Picard(d.size(1), max_iter=max_iter, tol=tol, whiten=whiten, ortho=ortho)
        result = torch.tensor(transform.fit_transform(d.cpu()), dtype=data.dtype, device=data.device).transpose(0, 1)
        results.append(result)
    return torch.stack(results)


def robust_scaler(data: torch.Tensor) -> torch.Tensor:
    scaled = RobustScaler().fit_transform(data.cpu().numpy().reshape(-1, data.shape[-1]))
    return torch.tensor(scaled.reshape(data.shape), dtype=data.dtype, device=data.device)


def ica_fit_and_transform(
    data: torch.Tensor,  # shape [N, C, T]
    train_mask: torch.Tensor,  # shape [N], boolean or long indices
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> torch.Tensor:
    """
    Learn a single ICA decomposition on all training-set trials,
    then apply that unmixing to the entire dataset.

    Args:
        data (torch.Tensor):
            EEG data of shape [num_trials, num_channels, num_timepoints].
        train_mask (torch.Tensor):
            Boolean or index mask of shape [num_trials] indicating which trials are training.
        max_iter (int):
            Maximum number of Picard iterations.
        tol (float):
            Tolerance for the stopping criterion in Picard.

    Returns:
        torch.Tensor:
            ICA-transformed data of shape [num_trials, num_channels, num_timepoints].
            Each trial has the same shape as input but in the ICA component space.
    """

    device = data.device
    dtype = data.dtype

    # ------------------------------------------
    # 1) EXTRACT & CONCATENATE ALL TRAINING TRIALS
    # ------------------------------------------
    # data[train_mask] has shape [n_train, C, T]
    train_data = data[train_mask]
    n_train, C, T = train_data.shape

    # We want shape [n_samples, n_features] = [n_train*T, C]
    # So permute to [n_train, T, C], then reshape
    # => shape [n_train*T, C]
    train_data_2d = train_data.permute(0, 2, 1).reshape(-1, C)

    # Move to CPU if necessary, because Picard typically runs on NumPy
    train_data_np = train_data_2d.cpu().numpy()

    # ------------------------------------------
    # 2) FIT ICA ON ALL TRAINING DATA
    # ------------------------------------------
    ica_transform = picard.Picard(n_components=C, max_iter=max_iter, tol=tol)
    ica_transform.fit(train_data_np)

    # ------------------------------------------
    # 3) APPLY ICA TO ALL TRIALS (TRAIN/VAL/TEST)
    # ------------------------------------------
    results = []
    N = data.shape[0]
    for i in range(N):
        # Each trial: shape [C, T]
        trial = data[i]
        # Transpose to [T, C] for transform()
        trial_t = trial.transpose(0, 1).cpu().numpy()

        # Transform with the learned unmixing
        trial_ica = ica_transform.transform(trial_t)
        # => shape [T, C]

        # Convert back to Torch, and transpose to [C, T] if desired
        trial_ica_torch = torch.tensor(trial_ica, dtype=dtype, device=device).transpose(0, 1)

        results.append(trial_ica_torch)

    # Stack back into [N, C, T]
    return torch.stack(results, dim=0)


# def bandpass_filter(data, low=2, high=40, rate=1024, order=10):
#     sos = scipy.signal.butter(order, [low / rate / 2, high / rate / 2], btype='band', output='sos')
#     return torch.tensor(scipy.signal.sosfilt(sos, np.ascontiguousarray(data.cpu())), dtype=data.dtype,
#                         device=data.device)


def bandpass_filter(
    data: torch.Tensor = None, low=2, high=40, rate=1024, order=100, axis=-1, zero_phase=True
) -> torch.Tensor:
    sos = scipy.signal.butter(order, [low / (rate / 2), high / (rate / 2)], btype="band", output="sos")

    if data is None:
        return sos

    np_data = np.ascontiguousarray(data.detach().cpu().numpy())

    if zero_phase:
        op = scipy.signal.sosfiltfilt
    else:
        op = scipy.signal.sosfilt

    filtered = op(sos, np_data, axis=axis)

    # Fix stride issues with filtfilt.
    filtered = filtered.copy()

    return torch.from_numpy(filtered).type_as(data).to(data.device)


def preprocess_data(
    data: torch.Tensor, labels: torch.Tensor, modality_idx: int, artifact_idx: int, max_stimuli_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:

    # Get imagined trials with vowels and no artifacts
    mask1 = labels[:, 0] == modality_idx
    mask2 = labels[:, 2] == artifact_idx
    mask3 = labels[:, 1] <= max_stimuli_idx
    mask = mask1 & mask2 & mask3

    data = common_average_referencing(normalize(data, dim=2), dim=1)
    labels = labels[:, 1]

    data = data[mask]
    labels = labels[mask]

    return data, labels


def get_preprocessed_dataset(logger: logging.Logger) -> tuple[torch.Tensor, torch.Tensor]:
    """Get the preprocessed data and labels.

    Returns:
        torch.Tensor: The preprocessed data.
        torch.Tensor: The labels.
    """
    data, labels, subjects, metadata = get_dataset(file_name="dataset.pt")

    logger.info(f"Original data shape: {data.shape}, labels shape: {labels.shape}, subjects shape: {subjects.shape}")

    modalities = metadata["modalities"]
    artifacts = metadata["artifacts"]
    stimuli = metadata["stimuli"]

    return preprocess_data(
        data=data,
        labels=labels,
        modality_idx=modalities.index("Imagined"),
        artifact_idx=artifacts.index("None"),
        max_stimuli_idx=stimuli.index("U"),
    )


if __name__ == "__main__":

    logger = get_logger(log_filename="preprocessing.log")

    data, labels, subjects, metadata = get_dataset(file_name="dataset.pt")

    logger.info(f"Original data shape: {data.shape}, labels shape: {labels.shape}, subjects shape: {subjects.shape}")

    modalities = metadata["modalities"]
    artifacts = metadata["artifacts"]
    stimuli = metadata["stimuli"]

    logger.info(f"\nModalities: {modalities},\nArtifacts: {artifacts},\nStimuli: {stimuli}")

    data, labels = preprocess_data(
        data=data,
        labels=labels,
        modality_idx=modalities.index("Imagined"),
        artifact_idx=artifacts.index("None"),
        max_stimuli_idx=stimuli.index("U"),
    )

    logger.info(f"Data shape after preprocessing: {data.shape}, labels shape: {labels.shape}")

    logger.info("Scaling data with robust scaler")
    data = robust_scaler(data)

    logger.info("Applying bandpass filter")
    data = bandpass_filter(data)

    logger.info(f"Data shape after bandpass filter: {data.shape}")
    logger.info("Applying Independent Component Analysis")

    # Using only one trial to save computation time
    components = ica(downsample(data)[100].unsqueeze(dim=0))
    logger.info(f"Components shape: {components.shape}\nComponents: {components[0]}")

    # Save components plot
    plt.figure(figsize=(15, 5))
    plt.imshow(components[0].numpy(), aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time Steps")
    plt.ylabel("Components")
    plt.title("Heatmap")
    plt.yticks(range(components.size(1)), [f"Component {i + 1}" for i in range(components.size(1))])
    plt.savefig(Config.plot_dir / "components_heatmap.png", transparent=True)
    plt.show()
