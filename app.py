import os
import time
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import nibabel as nib
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from flask import Flask, render_template, request, url_for

from model import BrainTumorClassifier2D, ensure_model_file
from database import (
    init_db,
    insert_patient,
    get_patient_history,
    get_all_patients,
    get_patient
)

# =========================================================
# FLASK CONFIGURATION
# =========================================================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp', 'nii', 'gz'}

init_db()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['no', 'yes']

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =========================================================
# LOAD AI MODEL SAFELY
# =========================================================
model_loaded = False
model_error_msg = ""
model = BrainTumorClassifier2D(num_classes=2).to(device)

model_path = "brain_model.pth"
# Ensure the model exists locally or download it from storage
ensure_model_file(model_path)

if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            
            if 'class_names' in checkpoint:
                CLASS_NAMES = checkpoint['class_names']
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        model_loaded = True
        print(f"✅ MODEL LOADED FROM {model_path}", flush=True)
        print(f"📌 Class Names: {CLASS_NAMES}", flush=True)
    except Exception as e:
        model_error_msg = str(e)
        print(f"❌ MODEL LOAD ERROR: {e}", flush=True)
else:
    model_error_msg = f"'{model_path}' not found in root directory."
    print(f"⚠️ WARNING: {model_error_msg}", flush=True)


def cleanup_static():
    current = time.time()
    for f in os.listdir(STATIC_FOLDER):
        path = os.path.join(STATIC_FOLDER, f)
        try:
            if os.path.isfile(path) and (current - os.path.getmtime(path) > 3600):
                os.remove(path)
        except Exception:
            pass


def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS or filename.lower().endswith('.nii.gz')


# =========================================================
# HIGH-SPEED GRAD-CAM & CENTROID EXTRACTION
# =========================================================
def generate_gradcam_mask(model, input_tensor, class_idx):
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    def save_activation(module, input, output):
        activations.append(output)

    target_layer = None
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layer4'):
        target_layer = model.backbone.layer4[-1]
    elif hasattr(model, 'layer4'):
        target_layer = model.layer4[-1]

    if target_layer is None:
        return None, None, None

    h1 = target_layer.register_forward_hook(save_activation)
    cam_input = input_tensor.clone().detach().requires_grad_(True)
    
    model.zero_grad()
    output = model(cam_input)

    if len(activations) > 0:
        activations[0].register_hook(save_gradient)

    score = output[0, class_idx]
    score.backward()
    h1.remove()

    if len(gradients) > 0 and len(activations) > 0:
        act = activations[0].detach()   # Shape: [1, C, H, W]
        grad = gradients[0].detach()    # Shape: [1, C, H, W]
        
        # Fast PyTorch GPU/CPU tensor pooling
        weights = torch.mean(grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * act, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        if np.max(cam) > 0:
            cam = cam / np.max(cam)

        y_indices, x_indices = np.where(cam > 0.50)
        if len(x_indices) > 0 and len(y_indices) > 0:
            center_x = int(np.mean(x_indices))
            center_y = int(np.mean(y_indices))
            return cam, center_x, center_y

    return None, None, None


# =========================================================
# ULTRA-FAST 3D SPATIAL VOLUMETRIC EXTENSION
# =========================================================
def create_brain_volume_from_2d(gray_2d, depth=24):
    """
    Constructs a 3D brain mesh using fast PyTorch affine grids.
    """
    h, w = gray_2d.shape
    gray_tensor = torch.from_numpy(gray_2d).unsqueeze(0).unsqueeze(0).float() / 255.0

    z_coords = np.linspace(-0.9, 0.9, depth)
    slices = []

    for z in z_coords:
        scale = max(0.1, np.sqrt(max(0.0, 1.0 - (z ** 2))))
        scaled = F.interpolate(gray_tensor, scale_factor=scale, mode='bilinear', align_corners=False)
        pad_top = (h - scaled.shape[2]) // 2
        pad_bottom = h - scaled.shape[2] - pad_top
        pad_left = (w - scaled.shape[3]) // 2
        pad_right = w - scaled.shape[3] - pad_left
        padded = F.pad(scaled, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        slices.append(padded.squeeze().numpy())

    volume = np.stack(slices, axis=-1)
    return gaussian_filter(volume, sigma=(0.8, 0.8, 0.8))


# =========================================================
# FAST RADIOLOGICAL HEATMAP OVERLAYS
# =========================================================
def generate_slices(highres_gray, mask_2d):
    images = []
    try:
        session_ts = int(time.time())
        fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#0B0F19")
        ax.set_facecolor("#0B0F19")

        ax.imshow(highres_gray, cmap='bone', origin='upper')

        if np.sum(mask_2d > 0.4) > 0:
            smoothed_mask = gaussian_filter(mask_2d, sigma=1.2)
            masked_cam = np.ma.masked_where(smoothed_mask < 0.40, smoothed_mask)
            
            ax.imshow(masked_cam, cmap='inferno', alpha=0.55, origin='upper')
            ax.contour(smoothed_mask > 0.55, colors='#38BDF8', linewidths=1.0, alpha=0.85)
            ax.contour(smoothed_mask > 0.70, colors='#FF0055', linewidths=1.5)

        ax.axis("off")
        plt.tight_layout(pad=0)

        fname = f"slice_{session_ts}_0.png"
        fpath = os.path.join(STATIC_FOLDER, fname)
        plt.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)

        images.append(fname)
    except Exception as e:
        print("❌ SLICE GENERATION ERROR:", e, flush=True)

    return images


# =========================================================
# FAST 3D HOLOGRAPHIC BRAIN & BEACON RENDERER
# =========================================================
def generate_3d_plot(volume_data, center_x, center_y):
    try:
        sub_vol = volume_data[::2, ::2, :]
        brain = (sub_vol - sub_vol.min()) / (sub_vol.max() - sub_vol.min() + 1e-8)
        x, y, z = np.mgrid[0:brain.shape[0], 0:brain.shape[1], 0:brain.shape[2]]

        fig = go.Figure()

        fig.add_trace(go.Isosurface(
            x=x.flatten(), y=y.flatten(), z=z.flatten(),
            value=brain.flatten(),
            isomin=0.20, isomax=0.85,
            surface_count=3,
            opacity=0.15,
            colorscale=[
                [0.0, "#0F172A"],
                [0.4, "#0284C7"],
                [0.8, "#38BDF8"],
                [1.0, "#E0F2FE"]
            ],
            caps=dict(x_show=False, y_show=False, z_show=False),
            lighting=dict(ambient=0.8, diffuse=0.8, specular=0.4, roughness=0.2),
            showscale=False,
            name="Brain Mesh"
        ))

        if center_x is not None and center_y is not None:
            mapped_x = float((center_x / 224.0) * brain.shape[0])
            mapped_y = float((center_y / 224.0) * brain.shape[1])
            mapped_z = float(brain.shape[2] * 0.50)

            fig.add_trace(go.Scatter3d(
                x=[mapped_x], y=[mapped_y], z=[mapped_z],
                mode='markers',
                marker=dict(size=12, color='#FF0055', symbol='diamond', line=dict(color='#FFFFFF', width=2)),
                name="Tumor Center"
            ))

            fig.add_trace(go.Scatter3d(
                x=[mapped_x], y=[mapped_y], z=[mapped_z],
                mode='markers',
                marker=dict(size=22, color='rgba(255, 0, 85, 0.25)', symbol='circle'),
                showlegend=False, hoverinfo='none'
            ))

            fig.add_trace(go.Scatter3d(
                x=[mapped_x - 4, mapped_x + 4, None, mapped_x, mapped_x, None, mapped_x, mapped_x],
                y=[mapped_y, mapped_y, None, mapped_y - 4, mapped_y + 4, None, mapped_y, mapped_y],
                z=[mapped_z, mapped_z, None, mapped_z, mapped_z, None, mapped_z - 4, mapped_z + 4],
                mode='lines',
                line=dict(color='#FF0055', width=3),
                showlegend=False, hoverinfo='none'
            ))

        fig.update_layout(
            paper_bgcolor="#0B0F19",
            plot_bgcolor="#0B0F19",
            scene=dict(
                bgcolor="#0B0F19",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.3), up=dict(x=0, y=0, z=1))
            ),
            margin=dict(l=0, r=0, t=10, b=0),
            height=480
        )

        filename = f"brain3d_{int(time.time())}.html"
        filepath = os.path.join(STATIC_FOLDER, filename)
        fig.write_html(filepath, include_plotlyjs='cdn', full_html=False)
        return filename
    except Exception as e:
        print("❌ 3D RENDER ERROR:", e, flush=True)
        return None


# =========================================================
# INFERENCE & CLASSIFICATION PIPELINE
# =========================================================
def process_mri_file(filepath):
    is_3d = filepath.lower().endswith(('.nii', '.nii.gz'))
    
    if not is_3d:
        pil_img = Image.open(filepath).convert('RGB')
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
        highres_gray = np.array(pil_img.convert('L').resize((224, 224)), dtype=np.float32)
        gray_small = np.array(pil_img.convert('L').resize((64, 64)), dtype=np.float32)
        volume_data = create_brain_volume_from_2d(gray_small, depth=24)
    else:
        nii_data = nib.load(filepath).get_fdata()
        mid_z = nii_data.shape[2] // 2
        mid_slice = nii_data[:, :, mid_z]
        pil_img = Image.fromarray((mid_slice / (mid_slice.max() + 1e-8) * 255).astype(np.uint8)).convert('RGB')
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
        highres_gray = np.array(pil_img.convert('L').resize((224, 224)), dtype=np.float32)
        volume_data = create_brain_volume_from_2d(np.array(pil_img.convert('L').resize((64, 64)), dtype=np.float32), depth=24)

    predicted_label = "No Tumor Detected"
    tumor_type = "None"
    confidence = 0.0
    class_probabilities = {}
    gradcam_2d = None
    center_x, center_y = None, None

    if model_loaded:
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
            class_idx = int(np.argmax(probs))
            confidence = float(probs[class_idx] * 100)
            raw_class = str(CLASS_NAMES[class_idx]).lower().strip()

            class_probabilities = {
                str(CLASS_NAMES[i]).capitalize(): round(float(probs[i] * 100), 2)
                for i in range(len(CLASS_NAMES))
            }

        if raw_class in ["no", "notumor", "normal", "none", "health"]:
            predicted_label = "No Tumor Detected"
            tumor_type = "None"
        else:
            predicted_label = "Tumor Detected"
            tumor_type = "Positive (Lesion Present)" if raw_class == "yes" else raw_class.capitalize()
            try:
                gradcam_2d, center_x, center_y = generate_gradcam_mask(model, input_tensor, class_idx)
            except Exception as e:
                print("⚠️ Grad-CAM computation error:", e, flush=True)

    if tumor_type == "None" or predicted_label == "No Tumor Detected":
        mask_2d = np.zeros((224, 224), dtype=np.float32)
        volume_val = 0.0
        risk = "Low"
        center_x, center_y = None, None
    else:
        if gradcam_2d is not None:
            mask_2d = gradcam_2d
        else:
            norm_img = (highres_gray - highres_gray.min()) / (highres_gray.max() - highres_gray.min() + 1e-8)
            mask_2d = (norm_img > 0.85).astype(np.float32)

        voxel_count = np.sum(mask_2d > 0.50)
        volume_val = round(float(voxel_count) * 0.0018, 2)
        if volume_val < 0.1 and voxel_count > 0:
            volume_val = 1.10

        if volume_val < 1.2:
            risk = "Low"
        elif volume_val < 2.5:
            risk = "Medium"
        else:
            risk = "High"

    return (
        predicted_label, tumor_type, risk, round(confidence, 2),
        volume_val, highres_gray, mask_2d, volume_data, center_x, center_y,
        class_probabilities
    )


def calculate_growth(name):
    history = get_patient_history(name)
    volumes = []
    for h in history:
        try:
            volumes.append(float(h[5] or 0))
        except Exception:
            continue
    return volumes[::-1] if volumes else [0]


# =========================================================
# ROUTES
# =========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    cleanup_static()

    result, risk, tumor_type = "", "", ""
    confidence, volume_val = 0, 0
    plot_file = None
    growth, slice_images = [], []
    name, age, gender, patient_id = "", "", ""
    class_probabilities = {}

    if request.method == "POST":
        try:
            patient_id = f"PAT-{int(time.time())}"
            name = request.form.get("name", "Unknown")
            age = request.form.get("age", "N/A")
            gender = request.form.get("gender", "N/A")

            file = request.files.get("file")
            if not file or file.filename == '':
                return "❌ NO FILE UPLOADED", 400

            filename = secure_filename(file.filename)
            if not allowed_file(filename):
                return "❌ UNSUPPORTED FILE FORMAT", 400

            filepath = os.path.join(UPLOAD_FOLDER, f"{int(time.time())}_{filename}")
            file.save(filepath)

            (
                result, tumor_type, risk, confidence, volume_val, 
                highres_gray, mask_2d, volume_data, center_x, center_y,
                class_probabilities
            ) = process_mri_file(filepath)

            slice_images = generate_slices(highres_gray, mask_2d)
            plot_file = generate_3d_plot(volume_data, center_x, center_y)
            growth = calculate_growth(name)

            insert_patient((
                patient_id, name, age, gender, volume_val, risk, confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))

        except Exception as e:
            print("❌ ROUTE ERROR:", e, flush=True)
            return f"ERROR: {e}", 500

    return render_template(
        "index.html",
        result=result, volume=volume_val, risk=risk, confidence=confidence,
        plot_file=plot_file, name=name, age=age, gender=gender,
        patient_id=patient_id, growth=growth, tumor_type=tumor_type,
        slice_images=slice_images, model_loaded=model_loaded,
        model_error_msg=model_error_msg, class_probabilities=class_probabilities
    )


@app.route("/analytics")
def analytics():
    data = get_all_patients() or []
    dates = [d[8] for d in data]
    volumes = [d[5] for d in data]
    confidence = [d[7] for d in data]
    risk_labels = ["Low", "Medium", "High"]
    risks = [
        sum(1 for d in data if d[6] == "Low"),
        sum(1 for d in data if d[6] == "Medium"),
        sum(1 for d in data if d[6] == "High")
    ]
    return render_template(
        "analytics.html",
        dates=dates, volumes=volumes, confidence=confidence,
        risk_labels=risk_labels, risks=risks
    )


@app.route("/patients")
def patients():
    data = get_all_patients() or []
    return render_template("patients.html", patients=data)


@app.route("/report/<patient_id>")
def report_view(patient_id):
    p = get_patient(patient_id)
    if not p:
        return "❌ PATIENT NOT FOUND", 404

    patient = {
        "name": p[2], "age": p[3], "gender": p[4],
        "volume": p[5], "risk": p[6], "confidence": p[7], "date": p[8],
        "tumor_type": "Auto-detected", "growth": 0
    }
    return render_template("report.html", patient=patient)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server running on port {port}", flush=True)
    app.run(host='0.0.0.0', port=port)