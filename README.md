# FormulIA1

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF?logo=ultralytics&logoColor=black)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)

<p align="center">
  <img src="assets/demo.gif" alt="FormulIA1 Tracking Demo" width="800"/>
</p>

FormulIA1 is a modular Computer Vision pipeline designed to extract physical trajectories and structural insights from Formula 1 broadcast video feeds. It leverages instance segmentation, restricted ego-motion homography, and optional monocular depth estimation to perform highly stable, centroid-based tracking and motion trail visualization on fast-paced racing footage.

> [!NOTE]
> **Work In Progress (v0.01)**: This project is in its very early stages. While the core tracking architecture is there, the inference results are far from perfect, and the team classifier is currently even worse.
> 
> I am actively working on building better datasets, manually labeling more edge cases, and refining the overall logic.
> 
> See the **Roadmap & TODO** section below for upcoming features.

## Key Features

*   **Instance Segmentation Tracking**: Utilizes YOLOv8-seg for pixel-accurate vehicle and track masking, eliminating bounding box jitter.
*   **Restricted Ego-Motion Stabilization**: Computes robust visual odometry by isolating optical flow calculations exclusively to the semantic road surface.
*   **Monocular Depth-SLAM**: Integrates Depth Anything V2 for experimental 3D camera space unprojection and rigid-body transformation tracking.
*   **Cascade Classification Architecture**: Isolates tracking and team identification into a two-stage process for optimized inference.
*   **Automated Data Engineering**: Includes utilities integrating SegFormer and Segment Anything Model (SAM) for automated road and vehicle dataset labeling.

## Prerequisites & Installation

Requires Python 3.10+ and a CUDA-capable GPU for real-time inference.

```bash
git clone https://github.com/Moloshow/formulia1.git
cd formulia1

# Install dependencies
pip install -r requirements.txt
```

## Architecture Philosophy

### 1. Two-Stage Cascade (Detection ➔ Classification)
Rather than training a single YOLO model to detect 10 different F1 teams simultaneously, FormulIA1 decouples the problem into two models:
*   **Model A (Segmentation/Detection)**: Trained solely on a single class (`f1_car`). It learns the structural shape of a Formula 1 car and is highly robust to scale, blur, and lighting.
*   **Model B (Classification)**: A lightweight classifier that analyzes tight crops of the detected cars to identify the specific team (e.g., Ferrari, Mercedes).
*   **The "Why"**: Scalability and maintainability. When a new F1 season starts with new car liveries, only the lightweight classifier (Model B) needs fine-tuning on a small set of images. You completely avoid the massive cost of re-annotating tens of thousands of complex bounding boxes or polygons for the detection model.

### 2. Auto-Labeling with Foundation Models
Manual pixel annotation for semantic segmentation is extremely time-consuming. To achieve pixel-perfect tracking and ego-motion masking without manual labor, this pipeline relies on **Data Engineering via Foundation Models**:
*   **NVIDIA SegFormer** is used zero-shot to extract the drivable surface (the track) mask, allowing optical flow to ignore TV HUDs and grandstands.
*   **Meta SAM (Segment Anything Model)** is used to automatically upgrade coarse bounding boxes into precise F1 car polygons.

## Datasets & Data Flow

This project relies on a cascading data engineering pipeline to transform a basic object detection dataset into complex instance segmentation and classification datasets.

**The Source Dataset (`f1_generic`)**
*   **Origin**: [F1 Car Recognition 2](https://universe.roboflow.com/yoav-fogel-yia3f/f1-car-recognition-2) by Yoav Fogel.
*   **License**: CC BY 4.0
*   **Structure**: Standard YOLO Object Detection format (Bounding Boxes).
*   **Classes**: 10 classes corresponding to the F1 teams (Ferrari, Mercedes, Red-Bull, etc.).

**Generated Datasets (Automated via `tools/`)**
Running the data engineering scripts transforms `f1_generic` into three distinct, specialized datasets:
1.  **`f1_detection`**: A pure bounding-box dataset where all 10 team classes are collapsed into a single `0: f1_car` class. Used as the spatial prompt for SAM.
2.  **`f1_classification`**: An image classification dataset (ImageNet folder structure) containing only tight crops of the cars, categorized by team. Used to train the Team Classifier model.
3.  **`f1_segmentation`**: The final dataset. Upgrades the `f1_detection` boxes into pixel-perfect polygons using SAM, and adds a `1: road` semantic mask using SegFormer. Used to train the main YOLOv8-seg model.

## Configuration

The pipeline relies on **custom-trained** YOLO models. You must first generate the datasets and train these models using the provided scripts (see Usage section), which will output the weights to the following default directories. You can also override these paths using CLI arguments if you store your weights elsewhere.

*   **Segmentation Model**: `runs/segment/f1_road_segmentation/weights/best.pt`
*   **Classification Model**: `runs/classify/f1_team_classifier/weights/best.pt`
*   **Depth Model**: Automatically fetched from Hugging Face (`depth-anything/Depth-Anything-V2-Small-hf`)

## Usage

### Inference Pipeline

Run the main orchestrator on a video source.

**Standard tracking using semantic segmentation:**
```bash
python run_pipeline.py --source videos/sample_f1.mp4 --output outputs/result.mp4
```

**Enable the cascade Team Classifier to identify specific F1 teams:**
```bash
python run_pipeline.py --source videos/sample_f1.mp4 --output outputs/result.mp4 --classify
```

**Enable experimental 3D Depth-SLAM tracking:**
```bash
python run_pipeline.py --source videos/sample_f1.mp4 --depth-slam
```

**Debug mode (visualize semantic road mask and freeze edge tracking):**
```bash
python run_pipeline.py --source videos/sample_f1.mp4 --show-mask --freeze-edges
```

### Data Engineering

To avoid manual pixel annotation, this project relies on an automated dataset generation pipeline. 

**Auto-label segmentation masks from raw bounding box datasets:**
*Context: Fuses SegFormer (to extract the racing track plane) and SAM (to extract perfect vehicle silhouettes from coarse bounding boxes) into YOLO segmentation labels.*
```bash
python tools/auto_label_road.py --input-dir datasets/f1_detection --output-dir datasets/f1_segmentation
```

**Verify and render generated semantic masks:**
*Context: Debugging utility that reads YOLO segmentation `.txt` labels and overlays them as colored polygons on the original images to ensure the automated labeling process succeeded.*
```bash
python tools/visualize_segmentation.py --dataset-dir datasets/f1_segmentation
```

**Prepare Two-Stage Cascade datasets:**
*Context: To train the Team Classifier, we must extract tightly cropped images of the F1 cars. This script splits a generic detection dataset into two parts: a pure detection dataset and a folder structure of cropped cars ready for image classification training.*
```bash
python tools/prepare_datasets.py --base-dir datasets/f1_generic --det-dir datasets/f1_detection --cls-dir datasets/f1_classification
```

### Training & Evaluation

**Train the YOLOv8 instance segmentation model:**
```bash
python training/train_segmentation.py --model yolov8m-seg.pt --data datasets/f1_segmentation/data.yaml --epochs 30 --batch 8
```

**Train the YOLOv8 image classification model (Team Identifier):**
```bash
python training/train_classifier.py --model yolov8n-cls.pt --data-dir datasets/f1_classification --epochs 20 --batch 16
```

**Run the automated test suite:**
```bash
pytest
```

## Project Structure

```text
formulia1/
├── run_pipeline.py
├── README.md
├── tests/
│   └── test_run_pipeline.py
├── tools/
│   ├── auto_label_road.py
│   ├── prepare_datasets.py
│   └── visualize_segmentation.py
└── training/
    ├── evaluate_model.py
    ├── train_classifier.py
    └── train_segmentation.py
```

## Current Limitations & Known Issues

Transparency on technical debt and current system boundaries:
- **Optical Flow Instability**: Fast whip-pans with heavy motion blur can cause the homography matrix to degrade if too few trackable corners are found on the road surface.
- **Auto-Labeling Artifacts**: The SAM auto-labeling pipeline struggles when F1 cars are heavily occluded by fences or other cars, sometimes merging two cars into a single polygon.
- **Track Surface Segmentation**: The current zero-shot road extraction using Cityscapes-trained SegFormer is not perfectly adapted to F1 circuits (struggles with kerbs and runoff areas). Future improvements require manual labeling or sourcing a dedicated GP circuit dataset.
- **Dataset Temporal Bias**: The current `f1_generic` dataset is based on the 2024 F1 season. Because team liveries and sponsors change annually, the Team Classifier will suffer severe accuracy drops if run on broadcast videos from previous or future seasons. Generalization may also drop in severe weather conditions (e.g., heavy rain spray) or low-light night races.

## Roadmap & TODO

- [x] **Phase 1: Stable Segmentation Tracking**: Mask-centroid tracking and restricted ego-motion to eliminate bounding-box jitter.
- [ ] **Phase 2: Dataset & Labeling Refinement**: Expand the dataset to include multiple F1 seasons (to resolve temporal bias) and manually label complex track surfaces (kerbs/run-offs) to fix zero-shot segmentation artifacts.
- [ ] **Phase 3: Edge-Case Tracking Logic**: Implement robust fallback mechanisms for tracking occlusions, handle fast whip-pan optical flow failures, and refine mask-merging artifacts to improve overall tracker stability.
- [ ] **Phase 4: Radar Mini-Map**: Projection of $(cx, cy)$ pixel coordinates onto a 2D satellite orthophoto using multi-point homography registration.
- [ ] **Phase 5: Telemetry Sync**: Integration with the `FastF1` API via Optical Character Recognition (OCR) syncing to overlay real-time throttle/brake telemetry on the tracked vehicles.
