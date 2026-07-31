import torch
from torch.utils.data import Dataset, DataLoader
from model import Conv1dAutoencoder


# ---------------------------------------------------
# Example dataset
# Replace this with your own implementation.
# ---------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        """
        data:
            numpy array or tensor of shape
            (num_samples, sequence_length)
            OR
            (num_samples, channels, sequence_length)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

        if self.data.ndim == 2:
            # Add channel dimension
            self.data = self.data.unsqueeze(1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------
# Training loop
# ---------------------------------------------------
def train(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
):
    model.train()

    total_loss = 0.0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        reconstruction = model(batch)

        loss = criterion(reconstruction, batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def main():

    # -------------------------------------------------
    # Example: replace with your own datasets
    # -------------------------------------------------
    #
    # Dataset A -> shape (N, 512)
    # Dataset B -> shape (N, 1024)
    # Dataset C -> shape (N, 2048)
    #
    # They can all use the same model.
    #
    import numpy as np

    sequence_length = 512

    train_data = np.random.randn(
        1000,
        sequence_length
    )

    dataset = TimeSeriesDataset(train_data)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = Conv1dAutoencoder(
        in_channels=1,
        latent_dim=128,
        base_channels=32,
        bottleneck_length=16,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.MSELoss()

    epochs = 50

    for epoch in range(epochs):

        loss = train(
            model,
            loader,
            optimizer,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {loss:.6f}"
        )

    torch.save(
        model.state_dict(),
        "conv1d_autoencoder.pt"
    )


if __name__ == "__main__":
    main()