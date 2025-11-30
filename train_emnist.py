import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
from torchvision.models import resnet18

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# -----------------------------
# 1. Transform (EMNIST style)
# -----------------------------
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 2. Load EMNIST Balanced
# -----------------------------
train_set = EMNIST(root="./data", split="balanced", train=True, download=True, transform=transform)
test_set  = EMNIST(root="./data", split="balanced", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

num_classes = 47   # EMNIST Balanced

# -----------------------------
# 3. Build ResNet-18
# -----------------------------
model = resnet18(weights=None)         # No pretraining on ImageNet
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 1 channel input
model.fc = nn.Linear(512, num_classes) # Output 47 classes
model.to(device)

# -----------------------------
# 4. Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -----------------------------
# 5. Training Loop
# -----------------------------
def train_epoch(model, loader):
    model.train()
    total, correct, total_loss = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # EMNIST FIX: rotate + mirror
        imgs = torch.rot90(imgs, k=3, dims=[2,3])
        imgs = torch.flip(imgs, dims=[2])

        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, pred = torch.max(output, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return correct / total * 100, total_loss / len(loader)


# -----------------------------
# 6. Testing
# -----------------------------
def test_model(model, loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # EMNIST FIX
            imgs = torch.rot90(imgs, k=3, dims=[2,3])
            imgs = torch.flip(imgs, dims=[2])

            output = model(imgs)
            _, pred = torch.max(output, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


# -----------------------------
# 7. TRAIN FOR 10 EPOCHS
# -----------------------------
for epoch in range(10):
    train_acc, train_loss = train_epoch(model, train_loader)
    test_acc = test_model(model, test_loader)

    print(f"Epoch {epoch+1}/10 | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Loss: {train_loss:.4f}")

# -----------------------------
# 8. SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "resnet18_emnist_balanced.pth")
print("Model saved as resnet18_emnist_balanced.pth")
