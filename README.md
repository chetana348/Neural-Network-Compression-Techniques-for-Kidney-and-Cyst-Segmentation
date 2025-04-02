# UNet++ Compression Techniques for Kidney and Cyst Segmentation in ADPKD

This repository contains the **official code implementation** of our paper:

**[UNet++ Compression Techniques for Kidney and Cyst Segmentation in Autosomal Dominant Polycystic Kidney Disease](https://abe-journal.org/issues/2024/03/22/841)**  
📄 *Published in Applied Biomedical Engineering, 2024*

---

## 📌 Overview

We propose a compression framework to improve the efficiency of deep learning models used for semantic segmentation in medical imaging, specifically targeting **UNet++** architectures applied to **ADPKD (Autosomal Dominant Polycystic Kidney Disease)**.

Our method introduces a novel **Progressive Probabilistic Channel Aggregation (Progressive pCA)** mechanism, which compresses both:
- Neural network weights
- Intermediate prediction maps

These compressions occur **dynamically during training**, enabling:
- Reduced GPU memory usage
- Faster training convergence
- Minimal performance degradation

> ✅ Works out-of-the-box on kidney and cyst segmentation tasks with annotated MRI scans from ADPKD patients.

---

## 🛠️ Environment

To run this code on a local machine, please ensure the following versions are used:

- `python` = 3.8.5  
- `keras` = 2.8.0  
- `tb-nightly` = 2.9.0a20220301  
- `tensorflow` = 2.8.0  
- `tensorflow-gpu` = 2.3.0  
- `cudatoolkit` = 11.0.221=h74a9793_0  
- `cudnn` = 8.2.1=cuda11.3_0  

> ⚠️ **Note:** The code is optimized for **TensorFlow GPU 2.3.0** with **CUDA 11.x** support. Compatibility between TensorFlow, CUDA, and cuDNN is critical for correct GPU execution.

### 🔧 Recommended Setup (with conda)

```bash
conda create -n adpkd_compression_env python=3.8.5
conda activate adpkd_compression_env
pip install keras==2.8.0 tb-nightly==2.9.0a20220301 tensorflow==2.8.0 tensorflow-gpu==2.3.0

