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
