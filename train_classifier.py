import os
import sys
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from collections import Counter

print("⏳ [1/5] Loading PyTorch and dependencies...", flush=True)
from model import BrainTumorClassifier2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Device selected: {device}", flush=True)

# High-accuracy medical image augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

class TransformedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def load_pretrained_encoder(model, pretrained_path="brain_tumor_2d_unet_best.pth"):
    """Loads encoder weights from the UNet checkpoint into the classifier model."""
    if not os.path.exists(pretrained_path):
        print(f"⚠️ Pretrained file '{pretrained_path}' not found. Training from scratch.")
        return model

    try:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        encoder_weights = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            if clean_k.startswith(('enc1', 'enc2', 'enc3', 'bottleneck')):
                if 'enc1.0.weight' in clean_k and v.shape[1] != 3:
                    v = v[:, :3, :, :]
                encoder_weights[clean_k] = v

        model.load_state_dict(encoder_weights, strict=False)
        print(f"✅ Transferred {len(encoder_weights)} pretrained encoder layers into classifier!", flush=True)
    except Exception as e:
        print(f"⚠️ Could not load pretrained weights: {e}. Training from scratch.", flush=True)

    return model

def train_model(data_dir="data", epochs=20, batch_size=16, lr=2e-4, save_path="brain_model.pth"):
    if not os.path.exists(data_dir):
        print(f"❌ Error: Folder '{data_dir}' not found.", flush=True)
        return

    print("⏳ [3/5] Indexing dataset images...", flush=True)
    full_dataset = datasets.ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    print(f"✅ Found {len(full_dataset)} images across classes: {class_names}", flush=True)

    if len(full_dataset) == 0:
        print("❌ 'data' folder is empty.", flush=True)
        return

    # Calculate class weights to handle imbalance
    class_counts = Counter([label for _, label in full_dataset.samples])
    total_samples = len(full_dataset)
    class_weights = [total_samples / max(1, class_counts[i]) for i in range(len(class_names))]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    class_weights_tensor = class_weights_tensor / class_weights_tensor.sum()

    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_data = TransformedDataset(train_subset, transform=data_transforms['train'])
    val_data = TransformedDataset(val_subset, transform=data_transforms['val'])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print("⏳ [4/5] Initializing neural network & loading pretrained weights...", flush=True)
    model = BrainTumorClassifier2D(in_channels=3, num_classes=len(class_names)).to(device)
    model = load_pretrained_encoder(model, "brain_tumor_2d_unet_best.pth")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Phase 1: Warm up classification head (3 epochs)
    print("🔒 Freezing encoder backbone for warmup phase (epochs 1-3)...", flush=True)
    for name, param in model.named_parameters():
        if not name.startswith("classifier"):
            param.requires_grad = False

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n🚀 [5/5] Starting Training Loop...\n" + "=" * 50, flush=True)

    for epoch in range(epochs):
        # Phase 2: Unfreeze all layers after epoch 3
        if epoch == 3:
            print("\n🔓 Unfreezing entire model for deep fine-tuning...", flush=True)
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - 3, eta_min=1e-6)

        t0 = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data).item()

        if epoch >= 3:
            scheduler.step()

        epoch_train_loss = running_loss / len(train_data)
        epoch_train_acc = (running_corrects / len(train_data)) * 100

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(torch.argmax(outputs, 1) == labels.data).item()

        epoch_val_loss = val_loss / len(val_data)
        epoch_val_acc = (val_corrects / len(val_data)) * 100
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d}/{epochs:02d} ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.1f}%", flush=True)

        if epoch_val_acc >= best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("=" * 50, flush=True)
    print(f"🎉 Training Finished! Best Validation Accuracy: {best_acc:.2f}%", flush=True)

    checkpoint = {
        'state_dict': best_model_wts,
        'class_names': class_names
    }
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved to: '{save_path}'", flush=True)

if __name__ == "__main__":
    train_model(data_dir="data", epochs=20, batch_size=16)