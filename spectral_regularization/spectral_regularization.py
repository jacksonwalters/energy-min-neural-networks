import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse

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
# Compute adjacency matrix for the entire network
# -------------------------
def global_adjacency_matrix(model):
    """
    Construct symmetric block adjacency matrix for the entire network.
    Fixed version with proper layer size calculations.
    """
    layers = [p for n, p in model.named_parameters() if p.ndim == 2]
    
    # Correct way: layer_sizes should be [input_size, hidden1_size, hidden2_size, ..., output_size]
    layer_sizes = [layers[0].shape[0]]  # Start with input size
    for layer in layers:
        layer_sizes.append(layer.shape[1])  # Add output size of each layer
    
    total_nodes = sum(layer_sizes)
    A = torch.zeros((total_nodes, total_nodes), device=layers[0].device)

    # Calculate cumulative offsets for each layer
    layer_offsets = [0]
    for size in layer_sizes[:-1]:
        layer_offsets.append(layer_offsets[-1] + size)

    # Place weight matrices between consecutive layers
    for i, W in enumerate(layers):
        rows, cols = W.shape
        row_start = layer_offsets[i]
        col_start = layer_offsets[i + 1]
        
        # Place block and its transpose for symmetry
        A[row_start:row_start + rows, col_start:col_start + cols] = W
        A[col_start:col_start + cols, row_start:row_start + rows] = W.T

    return A

# -------------------------
# Spectral energy function for a single layer
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
# Spectral energy function for the entire model
# -------------------------
def spectral_energy_global(model,method='layerwise'):
    total = 0.0
    for _, W in model.named_parameters():
        if W.ndim == 2:
            if method == 'layerwise':
                total += spectral_energy(W) # compute sum of absolute eigenvalues for layer W
            if method == 'svd_sum':
                sv = torch.linalg.svdvals(W) # Eigenvalues of [[0, W], [W^T, 0]] are ± singular values of W
                total += 2 * sv.sum()  # ± singular values
            if method == 'frobenius':
                total += torch.sum(W**2) # use Frobenius norm as a proxy for spectral norm
            if method == 'global_adjacency':
                A = global_adjacency_matrix(model).cpu().detach() # arrange layers into global adjacency matrix
                eigs = torch.linalg.eigvalsh(A) # eigenvalues of global adjacency matrix
                total += torch.sum(torch.abs(eigs)).item() # sum of absolute eigenvalues
    return total.item()

# -------------------------
# Training function (with loss logging)
# -------------------------
def train_and_eval(use_regularizer=False, mu=1e-4, epochs=10):
    model = SmallMLP(hidden=128).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    acc_history = []
    loss_history = []
    spectral_history = []

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            spectral_loss = spectral_energy_global(model,method=args.method)

            if use_regularizer:
                loss += mu * spectral_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        loss_history.append(avg_loss)

        # Log global spectral energy on CPU
        spectral_history.append(spectral_loss)

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
        print(f"Epoch {epoch}, loss={avg_loss:.4f}, acc={acc:.4f}, spectral_energy={spectral_loss:.4f}")

    return model, loss_history, acc_history, spectral_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with spectral regularization")
    parser.add_argument("--mu", type=float, default=1e-4, help="Regularization strength")
    parser.add_argument("--method", type=str, default='layerwise', help="Regularization type")
    args = parser.parse_args()

    print(f"Using mu = {args.mu}")
    print(f"Using method = {args.method}")

    # -------------------------
    # Run experiments
    # -------------------------
    print("=== Baseline ===")
    baseline_model, baseline_loss, baseline_acc, baseline_spec = train_and_eval(use_regularizer=False)

    print("\n=== With Spectral Energy Regularizer ===")
    reg_model, reg_loss, reg_acc, reg_spec = train_and_eval(use_regularizer=True, mu=args.mu)

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

    plt.figure(figsize=(12,6))
    plt.plot(reg_spec, label="w/ spectral regularization")
    plt.xlabel("Epoch")
    plt.ylabel("Spectral Energy")
    plt.title("Spectral energy over epochs")
    plt.legend()
    plt.show()