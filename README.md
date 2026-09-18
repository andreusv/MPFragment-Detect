# MPFragDetect: Microplastics Detection & Morphological Analysis

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![GUI](https://img.shields.io/badge/GUI-PyWebView-orange.svg)](https://pywebview.flowrl.com/)
[![Inference Engine](https://img.shields.io/badge/Inference-ONNX%20Runtime-blue.svg)](https://onnxruntime.ai/)

**MPFragDetect** is an end-to-end computer vision software and inference engine designed for automated detection, segmentation, quantification, color classification, and morphological profiling of microplastic fragments from high-resolution optical microscope and scanner images.

---

## 🌟 Key Features

- **Deep Learning Instance Segmentation**: Powered by an optimized Mask R-CNN model executed via ONNX Runtime for high throughput and cross-platform compatibility.
- **High-Resolution Sliced Inference (SAHI)**: Processes large high-res microscope slides using overlapping sliding windows ($1024 \times 1024$ tiles with configurable overlap) to accurately detect sub-millimeter particles without downscaling degradation.
- **Instance-Aware Weighted Boxes & Mask Fusion**: Memory-efficient clustering pipeline (`cluster_detections`) that merges overlapping tile detections and fuses binary segmentation masks directly per instance using score-weighted ROI coverage maps.
- **HSV Particle Color Classification**: Automated color characterization (`ColorClassifier`) that classifies each detected microplastic fragment into standard polymer color categories (*Transparent/White, Black/Dark, Red, Blue, Green, Yellow, Orange, Purple*) along with RGB and Hex values.
- **Batch Image Processing & Scan Carousel**: Load and analyze single or multiple microscope slides simultaneously in batch mode, complete with an interactive scan carousel selector in the GUI.
- **Morphological Profiling & Quantitative Analysis**:
  - **Metrics Computed**: Surface Area ($\mu m^2$), Perimeter ($\mu m$), Equivalent Circular Diameter ($\mu m$), Feret Maximum Diameter ($\mu m$), Aspect Ratio, Circularity Index, and Solidity.
  - **Size Binning**: Automatically classifies detected particles into microplastic size tiers (Micro, Meso, Macro).
- **Automatic Scale Bar Detection**: Built-in computer vision module (`ScaleBarDetector`) that scans image corners for microscope scale bar markings to auto-calibrate the pixel-to-micrometer ratio.
- **Modern Desktop GUI**: Built using PyWebView and a responsive web dashboard (FastAPI backend + Vanilla JS/CSS frontend) offering visualization overlays, interactive data tables with color dot badges, and CSV/JSON exports.
- **Standalone Packaging**: Includes PyInstaller build automation (`build_app.py`) to create standalone executable packages (`.app` on macOS / `.exe` on Windows).
- **Benchmarking & Optimization Tools**: Includes evaluation scripts (`onnx_inference.py` and `WBF_EVO.py`) for COCO metric calculation, confusion matrix analysis, and WBF threshold tuning.

---

## 📁 Repository Structure

```
MPFragDetect/
├── app.py                            # GUI Application Entry Point (FastAPI + PyWebView)
├── mp_fragment_engine.py              # Core Engine (Sliced Inference, WBF, Color Classifier & Scale Calibration)
├── build_app.py                       # PyInstaller executable builder script
├── MPFragment.spec                    # PyInstaller build spec file
├── onnx_inference.py                  # Standalone model evaluation & COCO metrics pipeline
├── WBF_EVO.py                         # Hyperparameter tuning script for WBF thresholds
├── fragment-Config.yaml               # Model & dataset configuration file
├── requirements.txt                   # Dependency requirements file
├── final_maskrcnn_fragments_model.onnx# Trained Mask R-CNN ONNX model weights (176 MB)
├── static/                            # Frontend UI assets (HTML5, CSS3, JS)
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── test/                              # Sample evaluation dataset & COCO annotations
│   ├── _annotations.coco.json
│   └── images/
└── .gitignore                         # Git exclusion rules
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+** installed on your system.
- Recommended virtual environment (`venv` or `conda`).

### 1. Installation & Environment Setup

Clone the repository and install the dependencies:

```bash
git clone https://github.com/andreusv/MPFragmentApp.git
cd MPFragmentApp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required Python packages
pip install -r requirements.txt
```

### 2. Model Weights (`final_maskrcnn_fragments_model.onnx`)

The application requires the ONNX model file `final_maskrcnn_fragments_model.onnx` located in the root directory.

> ⚠️ **Important GitHub File Size Limit Note**:
> GitHub enforces a strict **100 MB file limit** for standard Git commits. Because `final_maskrcnn_fragments_model.onnx` is ~176 MB:
> - **Option A (Recommended for Git)**: Use **[Git LFS (Large File Storage)](https://git-lfs.github.com/)** to track `.onnx` files before pushing:
>   ```bash
>   git lfs install
>   git lfs track "*.onnx"
>   git add .gitattributes
>   ```
> - **Option B**: Exclude `.onnx` files in `.gitignore` and host the model file externally (e.g., GitHub Release assets, Google Drive, HuggingFace) with a download link provided in the setup instructions.

---

## 💻 Usage

### Launch Desktop Application

Run `app.py` to start the backend server and open the desktop application window:

```bash
python app.py
```

### Build Standalone Desktop Software

To package the application into a standalone executable (`dist/MPFragment.app` or `dist/MPFragment/MPFragment.exe`):

```bash
python build_app.py
```

### Model Evaluation & COCO Metrics

To evaluate model performance against COCO-annotated test datasets:

```bash
python onnx_inference.py --config fragment-Config.yaml --coco-json test/_annotations.coco.json --img-dir test/images
```

### Optimize WBF Hyperparameters

To tune Weighted Boxes Fusion IoU and confidence score thresholds:

```bash
python WBF_EVO.py --config fragment-Config.yaml
```

---

## 🔬 Morphological & Color Analysis Details

The analysis engine in [mp_fragment_engine.py](file:///Users/andreusimovidal/Desktop/MPFragDetect/mp_fragment_engine.py) extracts key physical and visual descriptors for microplastics research:

- **Particle Color Class**: Extracted using HSV histogram sampling inside the segmented particle mask.
- **Area ($\mu m^2$)**: Calculated from pixel mask count scaled by the pixel-to-$\mu m$ calibration factor.
- **Perimeter ($\mu m$)**: Extracted using mask contour perimeter algorithms.
- **Equivalent Circular Diameter ($\mu m$)**: Diameter of a circle with equivalent surface area.
- **Feret Maximum Diameter ($\mu m$)**: Maximum caliper length of the binary mask particle projection.
- **Aspect Ratio**: Ratio of major axis length to minor axis length ($AR = \frac{\text{Major Axis}}{\text{Minor Axis}}$).
- **Circularity**: Shape compactness descriptor defined as $4\pi \times \frac{\text{Area}}{\text{Perimeter}^2}$.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
