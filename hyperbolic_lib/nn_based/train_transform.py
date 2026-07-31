import torch
import torch.nn as nn
import torch.optim as optim

from hyperbolic_lib.lib.lorentz.manifold import CustomLorentz


# Generate synthetic data with noise
def generate_data(manifold=None, num_samples=1000, dim=64):
    x = torch.rand((num_samples, dim)) * 30 - 15  # x values between -15 and 15
    ks = torch.rand(num_samples) * 10  # x values between 0 and 10
    ks = ks.unsqueeze(-1).clamp(min=0.2)

    if manifold is not None:
        y = manifold.calc_time(x)
    else:
        y = torch.sqrt(torch.norm(x, dim=-1, keepdim=True)**2+ks)

    return x, y


class NeuralTransform(nn.Module):
    def __init__(self, in_features, out_features):
        super(NeuralTransform, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# Training parameters
def train_model():
    # Generate and split data
    X, y = generate_data(num_samples=10000, dim=64)
    dataset = torch.utils.data.TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    batch_size = 1024
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # Initialize model, loss, and optimizer
    model = NeuralTransform(64, 64).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Training loop
    num_epochs = 500
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.cuda())
            loss = criterion(outputs[..., 0].unsqueeze(-1), targets.cuda())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs.cuda())
                val_loss += criterion(outputs[..., 0].unsqueeze(-1), targets.cuda()).item()

        # Print progress
        if (epoch + 1) % 50 == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

    return model


# Train and evaluate
model = train_model()

