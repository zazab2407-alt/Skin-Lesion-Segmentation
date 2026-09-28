# 🧠 Skin Lesion Segmentation

A Computer Vision project for **skin lesion segmentation** from dermoscopic images using Python and multiple image segmentation techniques.

The project implements a complete pipeline including **image preprocessing, segmentation, post-processing, feature extraction, performance evaluation, and failure-case analysis**.

---

## 📌 Project Overview

The main goal of this project is to compare different classical image segmentation techniques for detecting skin lesions and evaluate their performance using quantitative metrics.

The pipeline was tested on **15 dermoscopic images** from the ISIC 2016 dataset.

---

## 🔄 Pipeline

```text
Input Image
     ↓
Preprocessing
     ↓
Segmentation
     ↓
Post-Processing
     ↓
Feature Extraction
     ↓
Performance Evaluation
     ↓
Results & Failure Analysis
```

---

## 🖼️ Preprocessing

The preprocessing stage includes:

* Green channel extraction
* CLAHE contrast enhancement
* Non-Local Means denoising

These steps help improve image quality and make the lesion more suitable for segmentation.

---

## 🔬 Segmentation Methods

Four segmentation techniques were implemented and compared:

### 1. Otsu Thresholding

Automatically determines an optimal threshold to separate the foreground from the background.

### 2. K-Means Clustering

Uses unsupervised clustering to divide image pixels into two groups.

### 3. Region Growing

Starts from selected seed regions and expands the segmented area based on pixel similarity.

### 4. Marker-Controlled Watershed

Uses distance transformation and markers to separate regions and generate the final segmentation mask.

---

## 🧹 Post-Processing

After segmentation, several operations are applied to improve the masks:

* Morphological Opening
* Morphological Closing
* Hole Filling
* Small-object removal
* Connected-component analysis

---

## 📐 Feature Extraction

The system extracts shape-based features from the segmented lesion:

* **Area**
* **Perimeter**
* **Circularity**

Circularity is calculated using:

```text
Circularity = (4 × π × Area) / Perimeter²
```

---

## 📊 Performance Evaluation

Each method is evaluated using:

* **Dice Coefficient**
* **IoU (Jaccard Index)**
* **Sensitivity**
* **Specificity**

### Results on 15 Images

| Method             |   Dice |    IoU | Sensitivity | Specificity |
| ------------------ | -----: | -----: | ----------: | ----------: |
| Otsu Threshold     | 0.7730 | 0.6533 |      0.7784 |      0.9445 |
| K-Means Clustering | 0.7730 | 0.6533 |      0.7784 |      0.9445 |
| Region Growing     | 0.7495 | 0.6187 |      0.7866 |      0.9357 |
| Marker Watershed   | 0.7236 | 0.6064 |      0.8807 |      0.7960 |

> **Note:** These results are based on the 15-image experiment currently configured in the project.

---

## ⚠️ Failure Analysis

The project automatically identifies low-performing segmentation cases using a **Dice threshold of 0.5**.

For detected failure cases, the system stores:

* Image ID
* Segmentation method
* Dice score
* IoU
* Sensitivity
* Specificity
* Possible failure reason

The failure cases are exported to:

```text
failure_cases.json
```

---

## 📁 Output Files

The project automatically generates:

```text
results/
├── detailed_metrics.csv
├── summary_metrics.csv
├── failure_cases.json
└── figures/
    ├── ISIC_00000000.png
    ├── ISIC_00000001.png
    └── ...
```

The generated figures provide a visual comparison between:

* Original image
* Ground Truth
* Preprocessed image
* Otsu Threshold
* K-Means Clustering
* Region Growing
* Marker Watershed

---

## 🛠️ Technologies

* Python
* OpenCV
* NumPy
* SciPy
* Scikit-image
* Matplotlib
* CSV / JSON

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/zazab2407-alt/Skin-Lesion-Segmentation.git
cd Skin-Lesion-Segmentation
```

### 2. Install dependencies

```bash
pip install numpy opencv-python matplotlib scikit-image scipy
```

### 3. Run the project

```bash
python project.py
```

The current configuration processes the first **15 images** from the dataset.

---

## 📂 Dataset

This project uses the **ISIC 2016 Skin Lesion Dataset** for experimentation.

The dataset itself is not included in this repository.

Place the dataset folders in the project directory and update the paths in `project.py` if necessary.

---

## 🎯 Project Objectives

* Apply classical computer vision techniques to medical images.
* Compare different segmentation algorithms.
* Evaluate segmentation quality using multiple metrics.
* Extract shape-based lesion features.
* Automatically identify and analyze segmentation failures.
* Generate reproducible results and visual comparisons.

---

## 👨‍💻 Author

**Zeyad Tharwat**

Computer Science / Information Technology Student

---

## 📌 Project Repository

[Skin Lesion Segmentation — GitHub](https://github.com/zazab2407-alt/Skin-Lesion-Segmentation)
