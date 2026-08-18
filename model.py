import os
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_model_file(path="brain_model.pth"):
    pass

# ==========================================
# 1. Direct Sequential Blocks (Exact Key Match)
# ==========================================
def make_double_conv_2d(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=True),  # .0
        nn.BatchNorm2d(out_c),                                         # .1
        nn.ReLU(inplace=True),                                         # .2
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=True), # .3
        nn.BatchNorm2d(out_c),                                         # .4
        nn.ReLU(inplace=True)                                          # .5
    )

def make_double_conv_3d(in_c, out_c):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=True),  # .0
        nn.BatchNorm3d(out_c),                                         # .1
        nn.ReLU(inplace=True),                                         # .2
        nn.Conv3d(out_c, out_c, kernel_size=3, padding=1, bias=True), # .3
        nn.BatchNorm3d(out_c),                                         # .4
        nn.ReLU(inplace=True)                                          # .5
    )

# ==========================================
# 2. 2D U-Net Encoder Classifier
# ==========================================
class BrainTumorClassifier2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, **kwargs):
        super().__init__()
        self.enc1 = make_double_conv_2d(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = make_double_conv_2d(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = make_double_conv_2d(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = make_double_conv_2d(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        b = self.bottleneck(self.pool3(x3))
        pooled = self.global_pool(b)
        return self.classifier(pooled)

# ==========================================
# 3. 3D U-Net Encoder Classifier
# ==========================================
class BrainTumorClassifier3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, **kwargs):
        super().__init__()
        self.enc1 = make_double_conv_3d(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = make_double_conv_3d(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = make_double_conv_3d(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = make_double_conv_3d(128, 256)

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        b = self.bottleneck(self.pool3(x3))
        pooled = self.global_pool(b)
        return self.classifier(pooled)

# ==========================================
# 4. Accurate Dual Weight Loader
# ==========================================
def load_dual_models(path_2d="brain_tumor_2d_unet_best.pth", path_3d="brain_tumor_3d_unet_best.pth", device=DEVICE):
    model_2d = BrainTumorClassifier2D(in_channels=3, num_classes=2).to(device)
    model_3d = BrainTumorClassifier3D(in_channels=1, num_classes=2).to(device)

    def extract_weights(ckpt_path, in_c, model_ref):
        if not os.path.exists(ckpt_path):
            print(f"[!] Warning: File '{ckpt_path}' not found.", flush=True)
            return {}
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt
            
            target_keys = set(model_ref.state_dict().keys())
            extracted = {}

            for k, v in state_dict.items():
                clean_k = k.replace("module.", "")

                # Adapt input channel differences (e.g. 4-channel pretrained to 3-channel or 1-channel)
                if clean_k == "enc1.0.weight" and v.shape[1] != in_c:
                    v = v[:, :in_c, ...]

                if clean_k in target_keys:
                    extracted[clean_k] = v
                elif clean_k.startswith(('enc1', 'enc2', 'enc3', 'bottleneck', 'classifier')):
                    extracted[clean_k] = v

            return extracted
        except Exception as e:
            print(f"[!] Error loading {ckpt_path}: {e}", flush=True)
            return {}

    # Prefer fine-tuned brain_model.pth if already created, else use base 2D UNet
    primary_2d_path = "brain_model.pth" if os.path.exists("brain_model.pth") else path_2d
    w_2d = extract_weights(primary_2d_path, in_c=3, model_ref=model_2d)
    if w_2d:
        model_2d.load_state_dict(w_2d, strict=False)
        print(f"[+] Loaded {len(w_2d)} layers into 2D Classifier from '{primary_2d_path}'.", flush=True)

    w_3d = extract_weights(path_3d, in_c=1, model_ref=model_3d)
    if w_3d:
        model_3d.load_state_dict(w_3d, strict=False)
        print(f"[+] Loaded {len(w_3d)} layers into 3D Classifier from '{path_3d}'.", flush=True)

    model_2d.eval()
    model_3d.eval()
    return model_2d, model_3d