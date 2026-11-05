"""Lab 08 - Digit Recognition on MNIST using a CNN (PyTorch).
Run:  python lab08_mnist_cnn.py
First run downloads MNIST (~11 MB) into ./data (internet needed once).
Outputs: training log, test accuracy, mnist_cnn.pth, predictions.png
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1000)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)   # 28x28
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)    # -> 14x14
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)    # -> 7x7
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                            # softmax is inside the loss

model = CNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate():
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in test_dl:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
    return correct / len(test_ds)

EPOCHS = 3
for epoch in range(1, EPOCHS + 1):
    model.train(); total = 0
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {epoch}/{EPOCHS}  loss={total/len(train_dl):.4f}  test acc={evaluate():.4f}")

torch.save(model.state_dict(), "mnist_cnn.pth")

model.eval()
x, y = next(iter(test_dl))
pred = model(x.to(device)).argmax(1).cpu()
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(x[i][0], cmap="gray"); ax.axis("off")
    ax.set_title(f"pred {pred[i].item()} / true {y[i].item()}")
plt.tight_layout(); plt.savefig("predictions.png")
print("Saved mnist_cnn.pth and predictions.png")
