import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# =========================================================
# CONFIGURATION & AUTO-DOWNLOAD UTILITY
# =========================================================

MODEL_PATH = "brain_model.pth"
# REPLACE THIS WITH YOUR GOOGLE DRIVE FILE ID
GDRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE" 

def ensure_model_file(model_path=MODEL_PATH, file_id=GDRIVE_FILE_ID):
    """
    Downloads the model file from Google Drive if it doesn't exist locally.
    """
    if not os.path.exists(model_path):
        print(f"[*] '{model_path}' not found locally. Downloading from Google Drive...", flush=True)
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, model_path, quiet=False)
            print(f"[+] Successfully downloaded '{model_path}'!", flush=True)
        except Exception as e:
            print(f"[-] Error downloading model file: {e}", flush=True)
            print("[-] Please ensure 'gdown' is installed and your Google Drive file is public.")
    return os.path.exists(model_path)


# =========================================================
# 2D RESNET CLASSIFIER (BINARY: 'no', 'yes')
# =========================================================

class BrainTumorClassifier2D(nn.Module):
    """
    2D ResNet-18 Classifier for Brain Tumor MRI Binary Dataset.
    Classes: ['no', 'yes'] (num_classes=2)
    """
    def __init__(self, num_classes=2):
        super(BrainTumorClassifier2D, self).__init__()
        # Pretrained ResNet18 backbone
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# =========================================================
# 3D U-NET (FOR 3D VOLUMETRIC SEGMENTATION)
# =========================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )

    def forward(self, x):
        return self.block(x)


class Simple3DUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(Simple3DUNet, self).__init__()

        # ENCODER
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d(2)

        # BOTTLENECK
        self.bottleneck = DoubleConv(128, 256)

        # DECODER
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        # FINAL OUTPUT
        self.final = nn.Conv3d(32, out_channels, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        b = self.bottleneck(p3)

        u3 = self.up3(b)
        if u3.shape[2:] != e3.shape[2:]:
            u3 = F.interpolate(u3, size=e3.shape[2:], mode='trilinear', align_corners=False)
        u3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        if u2.shape[2:] != e2.shape[2:]:
            u2 = F.interpolate(u2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        u2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        if u1.shape[2:] != e1.shape[2:]:
            u1 = F.interpolate(u1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        u1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(u1)

        return self.final(d1)


# =========================================================
# MODEL LOADER & INFERENCE PIPELINE
# =========================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image preprocessing transform for 2D classification
transform_2d = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_classifier(model_path=MODEL_PATH, num_classes=2, device=DEVICE):
    """
    Safely loads the trained 2D classifier model weights.
    """
    ensure_model_file(model_path)
    
    model = BrainTumorClassifier2D(num_classes=num_classes)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # Handles both full checkpoint dicts and direct state dicts
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)
        print(f"[+] Loaded model weights from '{model_path}'")
    else:
        print(f"[!] Warning: Model file '{model_path}' not found. Running with uninitialized weights.")
        
    model.to(device)
    model.eval()
    return model


def predict_tumor(image_path, model, class_names=None):
    """
    Runs inference on a single 2D image file path.
    Returns: (predicted_class_name, confidence_percentage)
    """
    if class_names is None:
        class_names = ['no', 'yes']
        
    image = Image.open(image_path).convert('RGB')
    tensor = transform_2d(image).unsqueeze(0).to(DEVICE)
    
    with torch.inference_mode():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100.0
        
    return class_names[predicted_idx], round(confidence, 2)


# =========================================================
# SELF-TEST BLOCK
# =========================================================

if __name__ == "__main__":
    print(f"[*] Testing models on device: {DEVICE}")

    # Test 2D Classifier
    model2d = BrainTumorClassifier2D(num_classes=2).to(DEVICE)
    x2d = torch.randn(1, 3, 224, 224).to(DEVICE)
    out2d = model2d(x2d)
    print(" 2D Model Output Shape:", out2d.shape)

    # Test 3D U-Net
    model3d = Simple3DUNet(in_channels=1, out_channels=3).to(DEVICE)
    x3d = torch.randn(1, 1, 64, 64, 64).to(DEVICE)
    out3d = model3d(x3d)
    print(" 3D Model Output Shape:", out3d.shape)