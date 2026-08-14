import os
import sys
import copy
import time

print("⏳ [1/5] Loading PyTorch and dependencies...", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

print("⏳ [2/5] Loading model architecture...", flush=True)
from model import BrainTumorClassifier2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Device selected: {device}", flush=True)

# Preprocessing transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

def train_model(data_dir="data", epochs=10, batch_size=16, lr=1e-4, save_path="brain_model.pth"):
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

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_data = TransformedDataset(train_subset, transform=data_transforms['train'])
    val_data = TransformedDataset(val_subset, transform=data_transforms['val'])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)

    print("⏳ [4/5] Initializing neural network...", flush=True)
    model = BrainTumorClassifier2D(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("\n🚀 [5/5] Starting Training Loop...\n" + "=" * 50, flush=True)

    for epoch in range(epochs):
        t0 = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_train_loss = running_loss / len(train_data)
        epoch_train_acc = (running_corrects.double() / len(train_data)).item()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        epoch_val_loss = val_loss / len(val_data) if len(val_data) > 0 else 0
        epoch_val_acc = (val_corrects.double() / len(val_data)).item() if len(val_data) > 0 else 0
        elapsed = time.time() - t0

        print(f"Epoch {epoch+1:02d}/{epochs:02d} ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc*100:.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc*100:.1f}%", flush=True)

        if epoch_val_acc >= best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print("=" * 50, flush=True)
    print(f"🎉 Training Finished! Best Validation Accuracy: {best_acc*100:.2f}%", flush=True)

    checkpoint = {
        'state_dict': best_model_wts,
        'class_names': class_names
    }
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved to: '{save_path}'", flush=True)

if __name__ == "__main__":
    train_model(data_dir="data", epochs=10, batch_size=16)