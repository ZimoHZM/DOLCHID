# Dental-Odontogenic-Lesion CBCT and Histopathology Integrated Dataset (DOLCHID)

This repository provides the official documentation, metadata, and access instructions for the **DOLCHID** dataset — the first curated multi-modal dataset integrating **Cone-Beam Computed Tomography (CBCT)** imaging and corresponding **histopathology (H&E-stained)** slices for odontogenic lesion research.

> ⚠️ **Note:**  
> Due to the inclusion of human clinical imaging and diagnostic data, the raw CBCT and histopathology images are **not stored in this repository**. Access is granted via figshare (see below).

---

## 📘 Dataset Overview

DOLCHID includes paired CBCT volumes and histopathological images collected from confirmed odontogenic lesion cases.  
Four major lesion subtypes are included:

- **Ameloblastoma (AME)**
- **Dentigerous Cyst (DC)**
- **Radicular Cyst (RC)**
- **Odontogenic Keratocyst (KCOT)**

Each case includes:

- CBCT imaging data  
- H&E-stained histopathology slides  
- Expert-verified lesion subtype labels  
- Segmentation annotations (where applicable)

This dataset supports tasks such as:

- Lesion segmentation (CBCT and histopathology)
- Single-modal lesion classification
- Multi-modal fusion and cross-modal learning

---

## 📁 What This Repository Contains

This repository serves as the **official documentation hub** for the DOLCHID dataset. It contains:

### ✔️ Dataset Description  
Detailed explanation of dataset composition, modalities, and class definitions.

### ✔️ Data Access Instructions  
How to obtain access via Figshare.

### ✔️ Customised Code
The implementation of the multi-modal fusion models used for DOLCHID validation.

### ✔️ Citation Information  
Guidance on how to cite the dataset in academic work, including BibTeX and DOI link.

---

## 🔐 Accessing the Dataset

The dataset is deposited on Figshare:

**DOI:** https://doi.org/10.6084/m9.figshare.30156622

---

## 🧠 Feature Fusion Models

To facilitate **multi-modal learning** on DOLCHID, this repository includes two reference PyTorch implementations that operate on pre-extracted CBCT and histopathology feature vectors:

- [Feature Fusion Models/grid_feature_fusion.py](https://github.com/ZimoHZM/DOLCHID/blob/main/Feature%20Fusion%20Models/grid_feature_fusion.py)
- [Feature Fusion Models/clip_feature_fusion.py](https://github.com/ZimoHZM/DOLCHID/blob/main/Feature%20Fusion%20Models/clip_feature_fusion.py)

---

## 📄 Citation

If you use the DOLCHID dataset in your work, please consider citing:

DOLCHID:

- 'Zimo Huang, Tian Xia, Tianfu Wu, Bing Liu, Shengfu Huang, Lei Bi and Jinman Kim. Dental Odontogenic Lesion CBCT and Histopathology Integrated Dataset. figshare. 2025. https://doi.org/10.6084/m9.figshare.30156622.'

Paper:

- 

