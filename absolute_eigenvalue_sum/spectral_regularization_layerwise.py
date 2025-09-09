import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

device = 'mps' if torch.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# -------------------------
# Dataset
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_ds = datasets.MNIST('.', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('.', train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False)

# -------------------------
# Model
# -------------------------
class SmallMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(28*28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# -------------------------
# Spectral energy function
# -------------------------
def spectral_energy(W):
    if W.ndim != 2:
        return torch.tensor(0.0, device=W.device)
    if W.size(0) == W.size(1):
        S = 0.5 * (W + W.t())
        eigs = torch.linalg.eigvalsh(S)
        return torch.sum(torch.abs(eigs))
    else:
        # Gram trick → singular values
        S = W.t() @ W if W.size(1) < W.size(0) else W @ W.t()
        eigs = torch.linalg.eigvalsh(S)
        return torch.sum(torch.sqrt(torch.clamp(eigs, min=1e-12)))

# -------------------------
# Training function (with loss logging)
# -------------------------
def train_and_eval(use_regularizer=False, mu=1e-4, epochs=10):
    model = SmallMLP(hidden=128).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    acc_history = []
    loss_history = []
    spectral_history = {name: [] for name, p in model.named_parameters() if p.ndim==2}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if use_regularizer:
                spec_loss = 0.0
                for name, p in model.named_parameters():
                    if p.ndim==2:
                        spec_loss = spec_loss + spectral_energy(p)
                loss = loss + mu * spec_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # Log spectral energy per layer
        for name, p in model.named_parameters():
            if p.ndim==2:
                with torch.no_grad():
                    if p.size(0)==p.size(1):
                        S = 0.5*(p+p.t())
                        eigs = torch.linalg.eigvalsh(S)
                        spectral_history[name].append(eigs.abs().sum().item())
                    else:
                        S = p.t()@p if p.size(1)<p.size(0) else p@p.t()
                        eigs = torch.linalg.eigvalsh(S)
                        spectral_history[name].append(torch.sum(torch.sqrt(torch.clamp(eigs,1e-12))).item())

        # Eval accuracy
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in test_loader:
                x,y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds==y).sum().item()
                total += y.numel()
        acc = correct/total
        acc_history.append(acc)
        print(f"Epoch {epoch}, loss={avg_loss:.4f}, acc={acc:.4f}")

    return model, loss_history, acc_history, spectral_history


# -------------------------
# Run experiments
# -------------------------
print("=== Baseline ===")
baseline_model, baseline_loss, baseline_acc, baseline_spec = train_and_eval(use_regularizer=False)

print("\n=== With Spectral Energy Regularizer ===")
reg_model, reg_loss, reg_acc, reg_spec = train_and_eval(use_regularizer=True, mu=1e-3)

# -------------------------
# Plot results
# -------------------------
epochs = np.arange(1, len(baseline_acc)+1)

plt.figure(figsize=(12,4))

# Loss
plt.subplot(1,2,1)
plt.plot(epochs, baseline_loss, label="Baseline")
plt.plot(epochs, reg_loss, label="Spectral Reg (μ=1e-4)")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.title("Training Loss vs. Epoch")

# Accuracy
plt.subplot(1,2,2)
plt.plot(epochs, baseline_acc, label="Baseline")
plt.plot(epochs, reg_acc, label="Spectral Reg (μ=1e-4)")
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.legend()
plt.title("Test Accuracy vs. Epoch")

plt.tight_layout()
plt.show()

# -------------------------
# Extract spectra
# -------------------------
def layer_spectra(model):
    spectra = {}
    for name, p in model.named_parameters():
        if p.ndim == 2:
            with torch.no_grad():
                if p.size(0) == p.size(1):
                    S = 0.5*(p + p.t())
                else:
                    S = p.t() @ p if p.size(1) < p.size(0) else p @ p.t()
                eigs = torch.linalg.eigvalsh(S.cpu())
                spectra[name] = eigs.numpy()
    return spectra

baseline_spectra = layer_spectra(baseline_model)
reg_spectra      = layer_spectra(reg_model)

# -------------------------
# Plot eigenvalue histograms
# -------------------------

layers = list(baseline_spectra.keys())
num_layers = len(layers)
cols = 2
rows = math.ceil(num_layers / cols)

plt.figure(figsize=(cols*6, rows*4))

for i, layer in enumerate(layers, 1):
    plt.subplot(rows, cols, i)
    plt.hist(baseline_spectra[layer], bins=50, alpha=0.5, label="Baseline")
    plt.hist(reg_spectra[layer], bins=50, alpha=0.5, label="Spectral Reg")
    plt.xlabel("Eigenvalue")
    plt.ylabel("Count")
    plt.legend()
    plt.title(f"Eigenvalue spectrum: {layer}")

plt.tight_layout()
plt.show()

# -------------------------
# Plot spectral energy over epochs
layers = list(baseline_spectra.keys())  # or spectral_history.keys()

plt.figure(figsize=(12,6))
for i, layer in enumerate(layers):
    plt.plot(reg_spec[layer], label=f"{layer}")
plt.xlabel("Epoch")
plt.ylabel("Sum of |Eigenvalues|")
plt.title("Layer-wise spectral energy over epochs")
plt.legend()
plt.show()
