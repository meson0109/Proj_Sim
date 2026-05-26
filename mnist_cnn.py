"""MNIST CNN training, evaluation, prediction, and plotting helpers."""

from pathlib import Path
import random
import time

import torch
from PIL import Image


class LeNetMNIST(torch.nn.Module):
    """A compact CNN that reaches well above 95% on MNIST in a few epochs."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _require_torchvision():
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for MNIST loading.") from exc
    return datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_mnist_splits(root="../Project2/mnist_dataset", split=(5, 1, 1), seed=42):
    """Return train/val/test subsets using the requested 5:1:1 style split."""
    datasets, transforms = _require_torchvision()
    transform = transforms.Compose([transforms.ToTensor()])
    train_part = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    test_part = datasets.MNIST(root=root, train=False, download=False, transform=transform)
    full = torch.utils.data.ConcatDataset([train_part, test_part])
    total = len(full)
    train_n = total * split[0] // sum(split)
    val_n = total * split[1] // sum(split)
    test_n = total - train_n - val_n
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(full, [train_n, val_n, test_n], generator=generator)


def make_loaders(train_set, val_set, test_set, batch_size=128, num_workers=2):
    return {
        "train": torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        ),
        "val": torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "test": torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
    }


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.shape[0]
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.shape[0]
    return total_loss / total, total_correct / total


def train_model(model, loaders, epochs=5, lr=1e-3, device=None, weight_decay=1e-4):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.shape[0]
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total += x.shape[0]
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc = evaluate(model, loaders["val"], device=device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"epoch {epoch:02d}: train loss {train_loss:.4f}, train acc {train_acc:.4f}, "
            f"val loss {val_loss:.4f}, val acc {val_acc:.4f}"
        )
    history["train_seconds"] = time.time() - start_time
    return history


def load_grayscale_image(path, auto_invert=True):
    img = Image.open(path).convert("L").resize((28, 28))
    tensor = torch.tensor(list(img.getdata()), dtype=torch.float32).reshape(1, 28, 28) / 255.0
    if auto_invert and tensor.mean() > 0.5:
        tensor = 1.0 - tensor
    return tensor


@torch.no_grad()
def predict_image_folder(model, folder="./MNIST-Test", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    folder = Path(folder)
    image_paths = sorted(
        p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )
    model.eval().to(device)
    results = []
    for path in image_paths:
        x = load_grayscale_image(path).unsqueeze(0).to(device)
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
        results.append((path.name, pred))
    return results


def save_checkpoint(model, history, path="outputs/mnist_cnn.pt"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "history": history}, path)
    return path


def plot_loss_curves(history, save_path="outputs/mnist_loss.png"):
    import matplotlib.pyplot as plt

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("cross entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    return save_path


@torch.no_grad()
def plot_prediction_examples(model, dataset, save_path="outputs/mnist_examples.png", device=None, count=12):
    import matplotlib.pyplot as plt

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    indices = torch.randperm(len(dataset))[:count].tolist()
    cols = min(6, count)
    rows = (count + cols - 1) // cols
    plt.figure(figsize=(cols * 1.6, rows * 1.9))
    for i, idx in enumerate(indices, 1):
        x, y = dataset[idx]
        pred = int(model(x.unsqueeze(0).to(device)).argmax(dim=1).item())
        plt.subplot(rows, cols, i)
        plt.imshow(x.squeeze(0), cmap="gray")
        plt.title(f"label {int(y)} / pred {pred}", fontsize=9)
        plt.axis("off")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    return save_path
