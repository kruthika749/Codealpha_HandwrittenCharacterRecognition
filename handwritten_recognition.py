
"
HANDWRITTEN CHARACTER RECOGNITION

Features:
 - Train CNN on MNIST or EMNIST
 - Evaluate, confusion matrix, accuracy/loss plots, sample predictions
 - Predict single image(s)
 - Export to ONNX
 - Streamlit app for live upload & prediction (run with: streamlit run handwritten_full.py -- --mode serve)
 - EMNIST image rotation/hflip fixed automatically
 - CLI interface: train, eval, predict, serve, export

Requirements:
 pip install torch torchvision matplotlib scikit-learn pillow tqdm streamlit

"""
import argparse
import os
import math
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import itertools
from tqdm import tqdm

# Optional import for Streamlit UI (only needed when serving)
try:
    import streamlit as st
except Exception:
    st = None


# Model: CNN

class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


# Utilities
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_transforms(dataset: str = "mnist"):
    """
    Builds transforms. For EMNIST we fix rotation and horizontal flip.
    Note: rotation and flip applied BEFORE ToTensor.
    """
    if dataset == "emnist":
        # EMNIST images must be rotated -90 and horizontally flipped to match standard orientation
        return transforms.Compose([
            transforms.Lambda(lambda img: img.rotate(-90, expand=True)),
            transforms.Lambda(lambda img: ImageOps.mirror(img)),
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

def load_datasets(dataset: str = "mnist", emnist_split: str = "balanced", root: str = "."):
    transform = build_transforms(dataset)
    if dataset == "mnist":
        train_ds = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
        classes = [str(i) for i in range(10)]
    else:
        train_ds = torchvision.datasets.EMNIST(root=root, split=emnist_split, train=True, download=True, transform=transform)
        test_ds = torchvision.datasets.EMNIST(root=root, split=emnist_split, train=False, download=True, transform=transform)
        # torchvision's EMNIST may not have .classes in some versions; fallback to numeric labels
        classes = getattr(train_ds, "classes", None)
        if classes is None:
            # Make numeric string labels if classes not available
            classes = [str(i) for i in range(len(set(train_ds.targets.tolist())))]
    return train_ds, test_ds, classes


# Training & Evaluation

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: str,
          epochs: int = 8, lr: float = 1e-3, save_dir: str = "pytorch_models", checkpoint_name: str = "handwritten_cnn.pth"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ensure_dir(save_dir)
    history = {"train_loss": [], "val_acc": []}

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n+1):.4f}"})

        avg_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        val_acc = evaluate(model, val_loader, device)
        history["val_acc"].append(val_acc)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {avg_loss:.4f}  Validation Accuracy: {val_acc:.2f}%")

        # Save after each epoch (keeps best practices)
        checkpoint_path = os.path.join(save_dir, f"epoch{epoch+1}_" + checkpoint_name)
        torch.save(model.state_dict(), checkpoint_path)

    # Final save
    final_path = os.path.join(save_dir, checkpoint_name)
    torch.save(model.state_dict(), final_path)
    print("Final model saved to:", final_path)
    return history, final_path

def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    device = device
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


# Predict / Inference helpers

def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_image(model: nn.Module, image_path: str, device: str, transform=None, classes: Optional[List[str]] = None) -> Tuple[int, float]:
    img = Image.open(image_path).convert("L")
    if transform:
        img_t = transform(img).unsqueeze(0).to(device)
    else:
        default_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_t = default_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_t)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        label = pred.item()
        confidence = conf.item()
    readable = classes[label] if classes is not None and label < len(classes) else str(label)
    return label, confidence, readable


# Visualization: plots, confusion matrix

def plot_history(history: dict, out_dir: str = "reports"):
    ensure_dir(out_dir)
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "train_loss.png"))
    plt.close()

    plt.figure()
    plt.plot(history["val_acc"], label="val_acc")
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "val_acc.png"))
    plt.close()
    print("Saved training plots in", out_dir)

def confusion_matrix_and_samples(model: nn.Module, loader: DataLoader, classes: List[str], device: str, out_dir: str = "reports", max_samples: int = 32):
    ensure_dir(out_dir)
    model.eval()
    all_preds = []
    all_labels = []
    samples = []  # collect (img_np, true, pred)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            # store up to max_samples
            if len(samples) < max_samples:
                imgs_cpu = images.cpu()
                for i in range(min(len(imgs_cpu), max_samples - len(samples))):
                    samples.append((imgs_cpu[i].squeeze().numpy(), int(labels[i]), int(preds[i])))

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    fig.savefig(cm_path)
    plt.close(fig)
    print("Saved confusion matrix to", cm_path)

    # Save sample prediction grid
    n = min(len(samples), 16)
    cols = 4
    rows = math.ceil(n / cols)
    fig2 = plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for i in range(n):
        img_np, t, p = samples[i]
        ax = fig2.add_subplot(rows, cols, i + 1)
        ax.imshow(img_np, cmap='gray')
        title = f"GT: {classes[t] if t < len(classes) else t}\nP: {classes[p] if p < len(classes) else p}"
        ax.set_title(title)
        ax.axis('off')
    samples_path = os.path.join(out_dir, "sample_predictions.png")
    fig2.tight_layout()
    fig2.savefig(samples_path)
    plt.close(fig2)
    print("Saved sample predictions to", samples_path)


# ONNX ExporT

def export_onnx(model: nn.Module, checkpoint_path: str, device: str, num_classes: int, out_path: str = "pytorch_models/handwritten_cnn.onnx"):
    # Load checkpoint into model if checkpoint provided
    model.to(device)
    # Make dummy input: batch=1, 1x28x28
    dummy = torch.randn(1, 1, 28, 28, device=device)
    model.eval()
    torch.onnx.export(model, dummy, out_path, input_names=["input"], output_names=["output"], opset_version=11)
    print("Exported ONNX model to", out_path)

# ------------------------------
# Streamlit App
# ------------------------------
def streamlit_app(model: nn.Module, transform, classes: List[str], device: str):
    if st is None:
        raise RuntimeError("Streamlit is not installed. Install with `pip install streamlit`")

    st.title("Handwritten Character Recognition")
    st.write("Upload an image (PNG/JPG) or draw a character elsewhere and upload it.")

    uploaded = st.file_uploader("Upload a handwritten image", type=["png", "jpg", "jpeg"])
    checkpoint_status = st.empty()

    if uploaded is not None:
        img = Image.open(uploaded).convert("L")
        st.image(img, caption="Uploaded image", use_column_width=True)
        if st.button("Predict"):
            label, conf, readable = predict_image(model, uploaded, device, transform=transform, classes=classes)
            st.success(f"Prediction: {readable}  — confidence: {conf*100:.2f}%")

    st.markdown("---")
    st.write("Try sample images from test set below (if dataset available).")
    if st.button("Show sample predictions (from test loader)"):
        # This requires having a test loader which we may not have here; instruct the user
        st.info("Run evaluation locally to generate reports (confusion matrix & sample predictions).")

# ------------------------------
# Main CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
    "--mode",
    choices=["train", "eval", "predict", "serve", "export"],
    default="train"
)

    parser.add_argument("--dataset", choices=["mnist", "emnist"], default="emnist")
    parser.add_argument("--emnist_split", default="balanced")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--save_dir", default="pytorch_models")
    parser.add_argument("--checkpoint", default=os.path.join("pytorch_models", "handwritten_cnn.pth"))
    parser.add_argument("--image", default=None, help="Path to single image for prediction")
    parser.add_argument("--image_dir", default=None, help="Path to directory of images for batch prediction")
    parser.add_argument("--out_dir", default="reports")
    parser.add_argument("--onnx_out", default=os.path.join("pytorch_models", "handwritten_cnn.onnx"))
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA even if available")
    args = parser.parse_args()

    device = "cpu" if args.no_cuda else get_device()
    print("Using device:", device)

    # Load datasets when needed
    if args.mode in ("train", "eval"):
        train_ds, test_ds, classes = load_datasets(dataset=args.dataset, emnist_split=args.emnist_split)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

        num_classes = len(classes)
    else:
        # For predict/serve/export we still need classes to map labels (we try to load dataset to get classes)
        try:
            _, test_ds, classes = load_datasets(dataset=args.dataset, emnist_split=args.emnist_split)
            num_classes = len(classes)
        except Exception:
            classes = [str(i) for i in range(10)]
            num_classes = 10


    model = CNN(num_classes=num_classes)

    if args.mode == "train":
        print("Starting training...")
        history, saved_path = train(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, save_dir=args.save_dir)
        plot_history(history, out_dir=args.out_dir)
        confusion_matrix_and_samples(model, test_loader, classes, device, out_dir=args.out_dir)
        print("Training complete. Model saved to:", saved_path)
        print(f"Reports saved to: {args.out_dir}")

    elif args.mode == "eval":
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model = load_checkpoint(model, args.checkpoint, device)
        acc = evaluate(model, test_loader, device)
        print(f"Validation Accuracy: {acc:.2f}%")
        confusion_matrix_and_samples(model, test_loader, classes, device, out_dir=args.out_dir)

    elif args.mode == "predict":
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model = load_checkpoint(model, args.checkpoint, device)
        transform = build_transforms(args.dataset)
        if args.image:
            label, conf, readable = predict_image(model, args.image, device, transform=transform, classes=classes)
            print(f"Image: {args.image}  Predicted: {readable}  Confidence: {conf*100:.2f}%")
        elif args.image_dir:
            from glob import glob
            files = sorted([p for p in glob(os.path.join(args.image_dir, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))])
            for f in files:
                label, conf, readable = predict_image(model, f, device, transform=transform, classes=classes)
                print(f"{Path(f).name}\t->\t{readable}\t({conf*100:.2f}%)")
        else:
            print("No --image or --image_dir provided.")

    elif args.mode == "serve":
        # Run as streamlit component; Streamlit will re-run main, so guard UI under this path.
        # Load model & then call streamlit_app
        if not os.path.exists(args.checkpoint):
            print("Checkpoint not found. You can still run UI but predictions won't work until checkpoint available.")
        else:
            model = load_checkpoint(model, args.checkpoint, device)
        transform = build_transforms(args.dataset)
        # If streamlit not installed, error
        if st is None:
            raise RuntimeError("Streamlit not installed. Install with `pip install streamlit` and run:\nstreamlit run handwritten_full.py -- --mode serve")
        # Streamlit expects the script to be run by streamlit; this call will only happen inside streamlit runtime.
        streamlit_app(model, transform, classes, device)

    elif args.mode == "export":
        # Export to ONNX. Make sure checkpoint is loaded first.
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        model = load_checkpoint(model, args.checkpoint, device)
        export_onnx(model, args.checkpoint, device, num_classes=num_classes, out_path=args.onnx_out)

    else:
        raise ValueError("Unknown mode: " + args.mode)

if __name__ == "__main__":
    main()


