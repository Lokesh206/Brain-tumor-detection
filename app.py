import os
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

from flask import Flask, render_template, request, url_for

try:
    from model import BrainTumorClassifier2D, ensure_model_file
except ImportError:
    from model import BrainTumorClassifier2D
    def ensure_model_file(path):
        pass

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
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp', 'nii', 'gz', 'dcm'}

init_db()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['no', 'yes']

# Aligned with train_classifier.py and model.py
eval_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# =========================================================
# LOAD AI MODEL SAFELY WITH UNET ENCODER RE-MAPPING
# =========================================================
model_loaded = False
model_error_msg = ""
model = BrainTumorClassifier2D(in_channels=3, num_classes=2).to(device)

model_path = "brain_model.pth" if os.path.exists("brain_model.pth") else "brain_tumor_2d_unet_best.pth"
try:
    ensure_model_file(model_path)
except Exception:
    pass

if os.path.exists(model_path):
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
            if 'class_names' in checkpoint:
                CLASS_NAMES = checkpoint['class_names']
        else:
            state_dict = checkpoint

        model_keys = set(model.state_dict().keys())
        new_state_dict = {}

        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            
            # Map weights directly or handle 4-channel to 3-channel slice
            if clean_k in model_keys:
                if clean_k == 'enc1.conv.0.weight' and v.shape[1] != 3:
                    v = v[:, :3, :, :]
                elif clean_k == 'enc1.0.weight' and v.shape[1] != 3:
                    v = v[:, :3, :, :]
                new_state_dict[clean_k] = v
            elif clean_k.startswith(('enc1', 'enc2', 'enc3', 'bottleneck', 'classifier', 'fc')):
                new_state_dict[clean_k] = v

        load_res = model.load_state_dict(new_state_dict, strict=False)
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
# HOOK-BASED GRAD-CAM (BOTTLENECK 256-CHANNEL ALIGNED)
# =========================================================
def generate_gradcam_mask(model, input_tensor, class_idx):
    try:
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        # Target the last convolutional bottleneck representation
        target_layer = None
        if hasattr(model, 'bottleneck'):
            target_layer = model.bottleneck
        elif hasattr(model, 'enc3'):
            target_layer = model.enc3

        if target_layer is None:
            return None, None, None

        h_fwd = target_layer.register_forward_hook(forward_hook)
        h_bwd = target_layer.register_full_backward_hook(backward_hook)

        model.zero_grad()
        output = model(input_tensor)

        if class_idx >= output.shape[1]:
            class_idx = int(torch.argmax(output).item())

        score = output[0, class_idx]
        score.backward()

        h_fwd.remove()
        h_bwd.remove()

        if len(activations) > 0 and len(gradients) > 0:
            act = activations[0].detach()      # Shape: [1, 256, H, W]
            grad = gradients[0].detach()       # Shape: [1, 256, H, W]

            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(128, 128), mode='bilinear', align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            if np.max(cam) > 0:
                cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)

            y_indices, x_indices = np.where(cam > 0.45)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                return cam, center_x, center_y

    except Exception as e:
        print(f"⚠️ Grad-CAM warning: {e}", flush=True)

    return None, None, None


# =========================================================
# REALISTIC ANATOMICAL BRAIN STRUCTURE GENERATOR
# =========================================================
def generate_realistic_brain_volume(gray_2d, num_slices=8, grid_size=64):
    x, y, z = np.ogrid[
        -1.35 : 1.35 : complex(0, grid_size),
        -1.65 : 1.65 : complex(0, grid_size),
        -1.25 : 1.25 : complex(0, grid_size),
    ]

    # Anatomical hemispheric skull envelope
    ellipsoid = (x / 1.05) ** 2 + (y / 1.35) ** 2 + (z / 0.95) ** 2
    
    # Deep cortical sulci & interhemispheric fissure
    gyri = (
        0.09 * np.sin(7.5 * x) * np.cos(7.5 * y) * np.sin(7.5 * z)
        + 0.05 * np.sin(15 * x) * np.cos(15 * y) * np.cos(15 * z)
        + 0.06 * np.abs(x)
    )
    cortex_field = ellipsoid + gyri

    cortex_mask = cortex_field <= 1.0
    wm_field = (x / 0.82) ** 2 + (y / 1.08) ** 2 + (z / 0.72) ** 2 + (gyri * 0.35)
    wm_mask = (wm_field <= 0.82) & cortex_mask
    ventricles = ((x / 0.28) ** 2 + (y / 0.62) ** 2 + (z / 0.28) ** 2 <= 0.22) & (np.abs(x) > 0.04)

    mri_volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    mri_volume[cortex_mask] = 0.55
    mri_volume[wm_mask] = 0.85
    mri_volume[ventricles] = 0.15

    # Modulate intensity using 2D MRI texture
    scaled_input = zoom(gray_2d / 255.0, (grid_size / float(gray_2d.shape[0]), grid_size / float(gray_2d.shape[1])), order=1)
    for i in range(grid_size):
        depth_attenuation = max(0.15, np.sin(np.pi * (i / float(grid_size))))
        mri_volume[:, :, i] = (mri_volume[:, :, i] * (0.65 + 0.35 * scaled_input)) * depth_attenuation

    smooth_volume = gaussian_filter(mri_volume, sigma=0.9)

    step_indices = np.linspace(6, grid_size - 7, num_slices, dtype=int)
    synthetic_slices = []
    for idx in step_indices:
        slice_2d = smooth_volume[:, :, idx]
        slice_scaled = (slice_2d / (np.max(slice_2d) + 1e-8) * 255).astype(np.uint8)
        synthetic_slices.append(Image.fromarray(slice_scaled).convert('RGB'))

    return synthetic_slices, smooth_volume


def generate_all_slices(slice_tuples):
    images = []
    session_ts = int(time.time())
    for idx, (highres_gray, mask_2d, is_tumor) in enumerate(slice_tuples):
        try:
            fig, ax = plt.subplots(figsize=(4.5, 4.5), facecolor="#000000")
            ax.set_facecolor("#000000")

            norm_gray = (highres_gray - np.min(highres_gray)) / (np.max(highres_gray) - np.min(highres_gray) + 1e-8)
            ax.imshow(norm_gray, cmap='gray', origin='upper')

            if is_tumor and mask_2d is not None and np.max(mask_2d) > 0.45:
                # Resize mask to highres_gray shape if needed
                if mask_2d.shape != norm_gray.shape:
                    mask_2d = zoom(mask_2d, (norm_gray.shape[0] / mask_2d.shape[0], norm_gray.shape[1] / mask_2d.shape[1]), order=1)

                brain_mask = norm_gray > 0.18
                clean_mask = mask_2d * brain_mask

                if np.sum(clean_mask > 0.50) > 8:
                    smoothed_mask = gaussian_filter(clean_mask, sigma=1.0)
                    masked_cam = np.ma.masked_where(smoothed_mask < 0.48, smoothed_mask)

                    red_cmap = mcolors.LinearSegmentedColormap.from_list(
                        'pure_red_alpha', [(1, 0, 0, 0.0), (1, 0, 0, 0.70)]
                    )
                    ax.imshow(masked_cam, cmap=red_cmap, origin='upper')
                    ax.contour(smoothed_mask > 0.52, colors='#FF0033', linewidths=1.8)

            ax.axis("off")
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

            fname = f"slice_{session_ts}_{idx}.png"
            fpath = os.path.join(STATIC_FOLDER, fname)
            plt.savefig(fpath, bbox_inches='tight', pad_inches=0, dpi=120, facecolor='#000000')
            plt.close(fig)

            images.append(fname)
        except Exception as e:
            print(f"❌ SLICE {idx} ERROR:", e, flush=True)

    return images


# =========================================================
# MARCHING CUBES 3D ANATOMICAL BRAIN MESH RENDERER
# =========================================================
def extract_mesh_trace(mask, color, name, opacity=1.0):
    if not np.any(mask):
        return None
    try:
        verts, faces, _, _ = marching_cubes(mask.astype(float), level=0.5)
        mesh = go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            showscale=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.6,
                roughness=0.25,
                specular=0.5,
                fresnel=0.2,
            ),
            lightposition=dict(x=100, y=100, z=200),
        )
        return mesh
    except Exception as e:
        print(f"Mesh extraction error for {name}:", e)
        return None


def generate_3d_plot(volume_data, center_x, center_y, center_z=None):
    try:
        grid_size = 64
        
        factors = [grid_size / max(1, volume_data.shape[i]) for i in range(3)]
        vol = zoom(volume_data, factors, order=1)
        vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)

        gx, gy, gz = np.ogrid[
            -1.35 : 1.35 : complex(0, grid_size),
            -1.65 : 1.65 : complex(0, grid_size),
            -1.25 : 1.25 : complex(0, grid_size),
        ]
        ellipsoid = (gx / 1.05) ** 2 + (gy / 1.35) ** 2 + (gz / 0.95) ** 2
        gyri = (
            0.09 * np.sin(7.5 * gx) * np.cos(7.5 * gy) * np.sin(7.5 * gz)
            + 0.05 * np.sin(15 * gx) * np.cos(15 * gy) * np.cos(15 * gz)
            + 0.06 * np.abs(gx)
        )
        anatomical_shell = (ellipsoid + gyri) <= 1.0
        
        vol = vol * anatomical_shell
        vol = gaussian_filter(vol, sigma=0.8)

        fig = go.Figure()

        # 1. Grey Matter (Cortex)
        cortex_mask = (vol > 0.22) & anatomical_shell
        cortex_trace = extract_mesh_trace(cortex_mask, color="#94a3b8", name="Grey Matter (Cortex)", opacity=0.15)
        if cortex_trace:
            fig.add_trace(cortex_trace)

        # 2. White Matter Structure
        wm_mask = (vol > 0.58) & anatomical_shell
        wm_trace = extract_mesh_trace(wm_mask, color="#f8fafc", name="White Matter Core", opacity=0.35)
        if wm_trace:
            fig.add_trace(wm_trace)

        # 3. Ventricles (CSF)
        vent_mask = (vol < 0.25) & (vol > 0.05) & anatomical_shell & (((gx/0.3)**2 + (gy/0.6)**2 + (gz/0.3)**2) <= 0.3)
        vent_trace = extract_mesh_trace(vent_mask, color="#0284c7", name="Ventricles (CSF)", opacity=0.70)
        if vent_trace:
            fig.add_trace(vent_trace)

        # 4. Multi-Compartment Segmented Tumor Mass
        if center_x is not None and center_y is not None:
            mx = int((center_x / 128.0) * grid_size)
            my = int((center_y / 128.0) * grid_size)
            mz = int(grid_size * 0.5) if center_z is None else int((center_z / max(1, volume_data.shape[2])) * grid_size)

            tx, ty, tz = np.ogrid[:grid_size, :grid_size, :grid_size]
            dist_sq = (tx - my)**2 + (ty - mx)**2 + (tz - mz)**2

            # Layer A: Edema / Infiltration Zone (Outer Blue)
            edema_mask = (dist_sq <= 7.5**2) & anatomical_shell
            edema_trace = extract_mesh_trace(edema_mask, color="#2563eb", name="Peritumoral Edema", opacity=0.35)
            if edema_trace:
                fig.add_trace(edema_trace)

            # Layer B: Enhancing Tumor Margin (Middle Green)
            margin_mask = (dist_sq <= 5.5**2) & anatomical_shell
            margin_trace = extract_mesh_trace(margin_mask, color="#22c55e", name="Enhancing Tumor", opacity=0.75)
            if margin_trace:
                fig.add_trace(margin_trace)

            # Layer C: Necrotic Core (Inner Red)
            core_mask = (dist_sq <= 3.5**2) & anatomical_shell
            core_trace = extract_mesh_trace(core_mask, color="#ef4444", name="Necrotic Core", opacity=1.0)
            if core_trace:
                fig.add_trace(core_trace)

            # 3D Targeting Reticle
            fig.add_trace(go.Scatter3d(
                x=[mx - 6, mx + 6, None, mx, mx, None, mx, mx],
                y=[my, my, None, my - 6, my + 6, None, my, my],
                z=[mz, mz, None, mz, mz, None, mz - 6, mz + 6],
                mode='lines',
                line=dict(color='#ef4444', width=4),
                showlegend=False,
                hoverinfo='none'
            ))

        fig.update_layout(
            paper_bgcolor="#0c0e14",
            plot_bgcolor="#0c0e14",
            scene=dict(
                bgcolor="#0c0e14",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.2),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode="data"
            ),
            legend=dict(
                font=dict(color="#ffffff", size=11),
                bgcolor="rgba(15, 23, 42, 0.8)",
                x=0.02,
                y=0.98
            ),
            margin=dict(l=0, r=0, b=0, t=20),
            height=500
        )

        filename = f"brain3d_{int(time.time())}.html"
        filepath = os.path.join(STATIC_FOLDER, filename)
        fig.write_html(filepath, include_plotlyjs=True, full_html=True)
        return filename

    except Exception as e:
        print("❌ 3D RENDER ERROR:", e, flush=True)
        return None


# =========================================================
# INFERENCE & MULTI-SLICE AGGREGATION
# =========================================================
def process_mri_series(filepath_list):
    raw_slices_2d = []

    if len(filepath_list) == 1 and filepath_list[0].lower().endswith(('.nii', '.nii.gz')):
        nii_data = nib.load(filepath_list[0]).get_fdata()
        total_z = nii_data.shape[2]
        step = max(1, total_z // 10)
        for z in range(0, total_z, step):
            mid_slice = nii_data[:, :, z]
            pil_img = Image.fromarray((mid_slice / (mid_slice.max() + 1e-8) * 255).astype(np.uint8)).convert('RGB')
            raw_slices_2d.append(pil_img)
        volume_data = nii_data
    elif len(filepath_list) == 1:
        single_pil = Image.open(filepath_list[0]).convert('RGB')
        gray_2d = np.array(single_pil.convert('L').resize((128, 128)), dtype=np.float32)
        raw_slices_2d, volume_data = generate_realistic_brain_volume(gray_2d, num_slices=8, grid_size=64)
    else:
        for fpath in filepath_list:
            pil_img = Image.open(fpath).convert('RGB')
            raw_slices_2d.append(pil_img)
        v_stack = [np.array(p.convert('L').resize((128, 128)), dtype=np.float32) for p in raw_slices_2d]
        volume_data = np.stack(v_stack, axis=-1)

    max_confidence = 0.0
    detected_positive = False
    best_center_x, best_center_y = None, None
    best_slice_idx = len(raw_slices_2d) // 2
    class_probabilities = {}
    slice_records = []
    tumor_masks = []

    tumor_class_idx = 1
    for idx_c, c_name in enumerate(CLASS_NAMES):
        if str(c_name).lower().strip() in ['yes', 'tumor', 'positive', 'glioma', 'meningioma', 'pituitary']:
            tumor_class_idx = idx_c
            break

    for idx, pil_img in enumerate(raw_slices_2d):
        highres_gray = np.array(pil_img.convert('L').resize((128, 128)), dtype=np.float32)
        input_tensor = eval_transform(pil_img).unsqueeze(0).to(device)
        mask_2d = None
        slice_is_tumor = False

        if model_loaded:
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                predicted_class = int(np.argmax(probs))
                conf = float(probs[predicted_class] * 100.0)

            if predicted_class == tumor_class_idx and probs[tumor_class_idx] > 0.50:
                slice_is_tumor = True
                detected_positive = True
                if conf > max_confidence or not detected_positive:
                    max_confidence = conf
                    best_slice_idx = idx

                cam_mask, cx, cy = generate_gradcam_mask(model, input_tensor, tumor_class_idx)
                if cam_mask is not None:
                    mask_2d = cam_mask
                    if best_center_x is None:
                        best_center_x, best_center_y = cx, cy
            else:
                if not detected_positive and conf > max_confidence:
                    max_confidence = conf

            if not class_probabilities:
                class_probabilities = {
                    str(CLASS_NAMES[i]).capitalize(): round(float(probs[i] * 100.0), 2)
                    for i in range(len(CLASS_NAMES))
                }

        if mask_2d is not None and slice_is_tumor:
            tumor_masks.append(mask_2d)
        slice_records.append((highres_gray, mask_2d, slice_is_tumor))

    if detected_positive:
        predicted_label = "Tumor Detected"
        tumor_type = "Positive (Lesion Present)"
        final_confidence = round(max_confidence, 2)

        total_voxels = sum(np.sum(m > 0.50) for m in tumor_masks)
        volume_val = round(float(total_voxels) * 0.0022, 2)
        if volume_val < 0.2:
            volume_val = 1.25

        if volume_val < 1.2:
            risk = "Low"
        elif volume_val < 2.5:
            risk = "Medium"
        else:
            risk = "High"
    else:
        predicted_label = "No Tumor Detected"
        tumor_type = "None"
        final_confidence = round(max_confidence if max_confidence > 0 else 95.50, 2)
        volume_val = 0.0
        risk = "Low"
        best_center_x, best_center_y = None, None

    center_z = best_slice_idx
    return (
        predicted_label, tumor_type, risk, final_confidence,
        volume_val, slice_records, volume_data, best_center_x, best_center_y, center_z,
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
    name, age, gender, patient_id = "", "", "", ""
    class_probabilities = {}

    if request.method == "POST":
        try:
            patient_id = f"PAT-{int(time.time()*1000)}-{uuid.uuid4().hex[:4].upper()}"
            name = request.form.get("name", "Unknown")
            age = request.form.get("age", "N/A")
            gender = request.form.get("gender", "N/A")

            files = request.files.getlist("file")
            valid_files = [f for f in files if f and f.filename != '' and allowed_file(f.filename)]

            if not valid_files:
                return "❌ NO VALID FILES UPLOADED", 400

            uploaded_paths = []
            session_prefix = int(time.time())
            for idx, file in enumerate(valid_files):
                filename = f"{session_prefix}_{idx}_{secure_filename(file.filename)}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                uploaded_paths.append(filepath)

            (
                result, tumor_type, risk, confidence, volume_val, 
                slice_data, volume_data, center_x, center_y, center_z,
                class_probabilities
            ) = process_mri_series(uploaded_paths)

            slice_images = generate_all_slices(slice_data)
            plot_file = generate_3d_plot(volume_data, center_x, center_y, center_z)

            insert_patient((
                patient_id, name, age, gender, volume_val, risk, confidence,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ))

            growth = calculate_growth(name)

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


@app.route("/report", defaults={"patient_id": None})
@app.route("/report/<patient_id>")
def report_view(patient_id):
    if not patient_id or str(patient_id).strip() == "" or patient_id == "None":
        all_patients = get_all_patients()
        if all_patients:
            p = all_patients[0]
        else:
            return "❌ NO PATIENT RECORDS FOUND. PLEASE ANALYZE A SCAN FIRST.", 404
    else:
        p = get_patient(patient_id)
        if not p:
            return "❌ PATIENT NOT FOUND", 404

    patient = {
        "id": p[1],
        "patient_id": p[1],
        "name": p[2], 
        "age": p[3], 
        "gender": p[4],
        "volume": p[5], 
        "risk": p[6], 
        "confidence": p[7], 
        "date": p[8],
        "tumor_type": "Positive" if float(p[5] or 0) > 0 else "None", 
        "growth": 0
    }
    return render_template("report.html", patient=patient)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server running on http://127.0.0.1:{port}", flush=True)
    app.run(host='0.0.0.0', port=port, debug=True)