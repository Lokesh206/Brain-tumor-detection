# AI-Powered Brain Tumor Detection & 3D Volumetric Analysis

An end-to-end medical imaging and computer vision platform designed for detecting, localizing, and analyzing brain tumors from MRI scans. The application processes standard 2D slices as well as 3D neuroimaging files, provides visual model interpretability via **Grad-CAM**, reconstructs interactive **3D spatial brain volumes**, and manages patient clinical histories through a responsive web interface.

---

## 🌟 Key Features

* **Multi-Format MRI Ingestion:** Accepts 2D medical image formats (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`) and 3D volumetric neuroimaging files (`.nii`, `.nii.gz`).
* **Deep Learning Classification:** Built on a fine-tuned ResNet-18 architecture for binary tumor diagnosis (`no` vs `yes`).
* **Visual Explainability (Grad-CAM):** Computes layer activations and gradients to generate radiological heatmap overlays and boundary contours indicating lesion regions.
* **Interactive 3D Holographic Rendering:** Builds a 3D spatial isosurface volume of the brain with 3D beacon markers pointing directly to the tumor centroid using Plotly.
* **Patient History & Analytics:** Embedded SQLite database to register patients, track longitudinal tumor growth metrics, monitor risk levels, and generate structured diagnostic reports.
* **Cloud & Edge Ready:** Automatic model weight downloading utility via Google Drive and dynamic port binding for zero-friction hosting on Render, Hugging Face, or Docker.

---

## 📁 Repository Structure

```text
├── app.py                  # Main Flask application and diagnosis pipeline
├── model.py                # PyTorch architectures (ResNet-18, 3D U-Net) & loaders
├── database.py             # SQLite schemas and clinical records management
├── requirements.txt        # Python package dependencies
├── Procfile                # Process file for cloud deployments (Render/Heroku)
├── Dockerfile              # Container configuration (Hugging Face/Docker)
├── templates/              # Jinja2 HTML templates
│   ├── index.html          # Scan upload and real-time inference dashboard
│   ├── patients.html       # Patient directory
│   ├── patient_details.html# Individual patient records and past scans
│   ├── patient_records.html# Full database registry view
│   ├── register.html       # Patient registration form
│   ├── report.html         # Printable diagnostic scan report
│   └── analytics.html      # Longitudinal tracking and risk distributions
├── static/                 # CSS stylesheets, client scripts, and generated slices
├── uploads/                # Temporary storage for uploaded scans
└── data/                   # Dataset directory for training pipelines
