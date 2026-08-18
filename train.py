import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# Configuration
DATA_DIR = "data"  # Folder containing 'no' and 'yes' subfolders
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
SAVE_PATH = "brain_model.pth"

def main():
    print("=" * 50, flush=True)
    print("🚀 STARTING BRAIN TUMOR MODEL TRAINING", flush=True)
    print("=" * 50, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Hardware Accelerator: {device}", flush=True)

    if not os.path.exists(DATA_DIR):
        print(f"❌ ERROR: Dataset folder '{DATA_DIR}' not found in {os.getcwd()}", flush=True)
        print("Please ensure your dataset is structured like:\n  dataset/\n    ├── no/\n    └── yes/", flush=True)
        return

    # Data Transformations
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"[*] Loading images from '{DATA_DIR}'...", flush=True)
    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    class_names = dataset.classes
    print(f"[*] Detected Classes: {class_names}", flush=True)
    print(f"[*] Total Training Images: {len(dataset)}", flush=True)

    # Initialize Pretrained ResNet-18
    print("[*] Initializing ResNet-18 architecture...", flush=True)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    print("\n" + "=" * 50, flush=True)
    print("🧠 MODEL TRAINING IN PROGRESS", flush=True)
    print("=" * 50, flush=True)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = (correct.double() / total) * 100.0

        print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] ── Loss: {epoch_loss:.4f} ── Accuracy: {epoch_acc:.2f}%", flush=True)

    # Save Checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, SAVE_PATH)

    print("\n" + "=" * 50, flush=True)
    print(f"✅ MODEL TRAINING COMPLETE! Saved to '{SAVE_PATH}'", flush=True)
    print("=" * 50, flush=True)

if __name__ == '__main__':
    main()