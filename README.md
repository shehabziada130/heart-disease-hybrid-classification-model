# 🫀 Hybrid Computer Vision Model for Heart Disease Prediction

This project applies **Computer Vision (CV)** techniques to detect heart diseases from **Chest X-ray (CXR)** images using deep learning.  
It combines **DenseNet121** and **Vision Transformer (ViT)** in a **hybrid model** to leverage both local feature extraction and global context understanding.  

---

## 🚀 Why Hybrid (DenseNet + ViT)?
- **DenseNet121**: Excellent at learning *fine-grained local features* from images (edges, textures, abnormalities).  
- **Vision Transformer (ViT)**: Captures *global relationships* across the image, making it suitable for identifying larger-scale abnormalities like cardiomegaly (enlarged heart).  
- **Hybrid Model**: Concatenates DenseNet and ViT features → achieving stronger performance than either model alone.  

This design balances **CNN’s local detail focus** with **Transformer’s holistic view**, which is crucial for medical imaging tasks.

---

## 🎯 Objectives
The project’s goal is to **automate heart disease detection from CXRs**, helping radiologists by:  
- Detecting abnormalities such as enlarged heart chambers, fluid buildup, and coronary artery disease.  
- Segmenting regions of interest like lungs and heart.  
- Classifying CXRs into disease categories (multi-label classification).  
- Reducing diagnosis time and minimizing human error.  

---

## 📂 Dataset
- Dataset: **NIH ChestX-ray14 (224x224 resized)** from Kaggle.  
- 14 disease classes including Atelectasis, Cardiomegaly, Effusion, Pneumonia, Nodule, and more.  
- Preprocessing:  
  - Removed “No Finding” labels.  
  - Split into **train (64%) / val (16%) / test (20%)**.  
  - Applied **Albumentations** for normalization and tensor conversion.  

---

## 🛠️ Tech Stack
- **Framework**: PyTorch  
- **Models**: DenseNet121 + Vision Transformer (ViT)  
- **Data Augmentation**: Albumentations  
- **Evaluation**: AUROC (per-class + mean), ROC curves  
- **Other Tools**: scikit-learn, matplotlib, tqdm, PrettyTable  

---

## 📊 Experiments & Results
- Training: 20 epochs, Adam optimizer, BCE loss, ReduceLROnPlateau scheduler  
- Evaluation metric: Mean AUROC  

**Results:**  
- **Train Mean AUROC**: `0.84`  
- **Test Mean AUROC**: `0.82`  

These results show strong generalization and reliability of the hybrid model for medical imaging tasks.  

### Example ROC Curve
*(`ROC train.png` / `ROC test.png`)*  
