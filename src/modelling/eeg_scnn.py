# -*- coding: utf-8 -*-
"""EGG Spiking Convolutional Neural Network."""
import copy
from typing import Type

import torch
import snntorch
import snntorch.functional
import snntorch.utils
import snntorch.surrogate
import torchmetrics
from tqdm import tqdm
from matplotlib import pyplot


from common.log import get_logger
from common.config import Config
from common.preprocessing import get_preprocessed_dataset


class EEGCSNN(torch.nn.Module):
    def __init__(
        self,
        beta: float,
        num_timesteps: int,
        out_features: int,
        in_channels: int,
        lr: float,
        loss_function: Type[snntorch.functional.LossFunctions],
        optimizer: Type[torch.optim.Optimizer],
        loss_kwargs: dict = None,
        opt_kwargs: dict = None,
    ):
        super().__init__()

        spike_grad = snntorch.surrogate.atan(alpha=2.0)

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),
            snntorch.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(kernel_size=2, padding=0, stride=2),
            snntorch.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(in_features=16384, out_features=out_features),
            # beta: randomly initialize decay rate for output neuron
            # Setting learn_beta=True enables the decay rate beta to be a learnable parameter
            snntorch.Leaky(beta=torch.rand(1), threshold=1.0, learn_beta=True, spike_grad=spike_grad, output=True),
        )
        self.num_timesteps = num_timesteps

        if loss_kwargs is None:
            loss_kwargs = {}

        if opt_kwargs is None:
            opt_kwargs = {}

        self.loss_function = loss_function(**loss_kwargs)
        self.optimizer = optimizer(self.net.parameters(), lr=lr, **opt_kwargs)  # noqa
        self._acc = torchmetrics.Accuracy(task="multiclass", num_classes=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden states at t=0
        snntorch.utils.reset(self.net)

        # Record for outputs
        spike_rec = []

        for step in range(self.num_timesteps):
            _, spike_out = self.net(x)
            spike_rec.append(spike_out)

        return torch.stack(spike_rec, dim=0)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> (torch.nn.Module, float):
        # X = self(X.reshape(X.shape[0], X.shape[1], X.shape[2]))
        X = self(X.unsqueeze(dim=1))

        accuracy = self._acc(X.sum(0).argmax(1), y)
        loss = self.loss_function(X, y)

        return loss, accuracy.item()

    def trainin_step(self, X: torch.Tensor, y: torch.Tensor) -> (float, float):
        loss, accuracy = self.evaluate(X=X, y=y)

        # Backpropagation
        loss.backward()

        # Update parameters
        self.optimizer.step()

        # Clear gradients
        self.optimizer.zero_grad()

        return loss.item(), accuracy

    def valid_or_test_step(self, X: torch.Tensor, y: torch.Tensor) -> (float, float):
        loss, accuracy = self.evaluate(X=X, y=y)
        return loss.item(), accuracy


def get_loader_from_dataset(
    dataset: torch.utils.data.Dataset, batch_size: int, suffle: bool = True
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=suffle)


def train():
    # Variables
    batch_size = 48
    num_epochs = 30
    # Number of epochs to wait for improvement before stopping
    early_stopping_patience = 5
    model_path = Config.model_dir / "eeg_scnn.pt"
    logger = get_logger(log_filename="train.log")

    # Make dirs if they don't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)
    Config.plot_dir.mkdir(parents=True, exist_ok=True)

    # Load the data
    data, labels = get_preprocessed_dataset(logger=logger)

    dataset = torch.utils.data.TensorDataset(data, labels)

    # Split dataset between train, validation and test
    num_samples = len(dataset)
    num_val = int(num_samples * 0.4)
    num_train = num_samples - num_val
    train_dataset, temp_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    valid_dataset, test_dataset = torch.utils.data.random_split(temp_dataset, [num_val // 2, num_val // 2])

    # Create the loaders
    train_loader = get_loader_from_dataset(train_dataset, batch_size=batch_size)
    val_loader = get_loader_from_dataset(valid_dataset, batch_size=batch_size)
    test_loader = get_loader_from_dataset(test_dataset, batch_size=batch_size)

    logger.info(
        f"Train dataset size: {len(train_dataset)}, Valid dataset size: {len(valid_dataset)}, Test dataset size: {len(test_dataset)}"
    )
    logger.info(f"Sample batch of train data: {next(iter(train_loader))[0].shape}")

    # Model configuration
    # Train SNN classifier
    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # network
    net = EEGCSNN(
        beta=0.95,
        num_timesteps=30,
        in_channels=1,
        out_features=5,
        loss_function=snntorch.functional.ce_count_loss,  # noqa
        loss_kwargs={},
        optimizer=torch.optim.Adam,
        opt_kwargs={"betas": (0.875, 0.95), "weight_decay": 0.1},
        lr=3e-3,
    )
    net.to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(net.optimizer, step_size=10, gamma=0.1)

    dtype = torch.float32
    n_train_batches = len(iter(train_loader))
    n_val_batches = len(val_loader)

    # Initialize the best model parameters and best validation acc
    best_model_params = copy.deepcopy(net.state_dict())
    best_val_acc = 0
    epochs_since_improvement = 0

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        val_acc = 0
        val_loss = 0
        train_batch = iter(train_loader)
        val_batch = iter(val_loader)

        # Minibatch training loop
        with tqdm(
            total=n_train_batches + n_val_batches, desc=f"Epoch {epoch}", unit=" batches", leave=True, position=0
        ) as pbar:
            for data, targets in train_batch:
                data = data.to(device)
                targets = targets.to(device)

                # forward pass and optimization
                net.train()
                loss, acc = net.trainin_step(X=data, y=targets)

                train_loss += loss
                train_acc += acc

                # Update progress bar for batches
                pbar.update(1)
                pbar.set_postfix(train_loss=loss)

            # Record average train accuracy and loss per epoch
            epoch_train_loss = train_loss / n_train_batches
            epoch_train_acc = train_acc / n_train_batches
            train_acc_hist.append(epoch_train_acc)
            train_loss_hist.append(epoch_train_loss)

            with torch.no_grad():
                for data, targets in val_batch:
                    data = data.to(device)
                    targets = targets.to(device)

                    # forward pass and optimization
                    net.eval()
                    _loss, _acc = net.valid_or_test_step(X=data, y=targets)

                    val_loss += _loss
                    val_acc += _acc

                    # Update progress bar for batches
                    pbar.update(1)
                    pbar.set_postfix(
                        train_loss=epoch_train_loss,
                        train_acc=epoch_train_acc,
                        val_loss=_loss,
                    )

            # Record average validation accuracy and loss per epoch
            epoch_val_loss = val_loss / n_val_batches
            epoch_val_acc = val_acc / n_val_batches
            val_acc_hist.append(epoch_val_acc)
            val_loss_hist.append(epoch_val_loss)
            pbar.update(0)
            pbar.set_postfix(
                train_loss=epoch_train_loss,
                train_acc=epoch_train_acc,
                val_loss=epoch_val_loss,
                val_acc=epoch_val_acc,
            )

            # Update the best model parameters if the current validation acc is better
            if epoch_val_acc > best_val_acc:
                best_val_acc = epoch_val_acc
                best_model_params = copy.deepcopy(net.state_dict())
                # Reset the counter if there is an improvement
                epochs_since_improvement = 0
            else:
                # Increment the counter if there is no improvement
                epochs_since_improvement += 1

        # Step scheduler
        scheduler.step()

        # Check for early stopping
        if epochs_since_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs with acc: {best_val_acc:.4f}")
            break

    # Load the best model parameters
    net.load_state_dict(best_model_params)
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    print("Saving model...")
    torch.save(net.state_dict(), model_path)

    # Plot loss curve
    pyplot.plot(range(len(train_loss_hist)), train_loss_hist, label="train loss")
    pyplot.plot(range(len(val_loss_hist)), val_loss_hist, label="validation loss")
    pyplot.title("Loss curve")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Loss")
    pyplot.legend()
    pyplot.savefig(Config.plot_dir / "loss.png")
    pyplot.show()

    # Plot loss curve
    pyplot.plot(range(len(train_acc_hist)), train_acc_hist, label="train acc")
    pyplot.plot(range(len(val_acc_hist)), val_acc_hist, label="val acc")
    pyplot.title("Accuracy")
    pyplot.xlabel("Epoch")
    pyplot.ylabel("Acc")
    pyplot.legend()
    pyplot.savefig(Config.plot_dir / "accuracy.png")
    pyplot.show()

    # Test the model
    test_batch = iter(test_loader)


if __name__ == "__main__":
    train()
