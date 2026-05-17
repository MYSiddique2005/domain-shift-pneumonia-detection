# Domain Shift Analysis in Deep CNN Models for Pneumonia Detection

A comparative study on cross-dataset generalization in deep learning-based pneumonia detection using chest X-ray imaging.

This project evaluates the robustness of VGG16, ResNet50, and DenseNet121 under domain shift conditions using the Kaggle Chest X-ray and RSNA Pneumonia Detection datasets. The work focuses on cross-domain performance degradation, model generalization, and Grad-CAM based interpretability analysis for medical imaging AI systems.

---

## Overview

Most deep learning research in pneumonia detection evaluates models on the same dataset used during training. While these approaches achieve high in-domain accuracy, they often fail to generalize across datasets collected from different hospitals or imaging environments.

This project investigates:

- Cross-dataset robustness of CNN architectures
- Impact of domain shift on pneumonia classification
- Model interpretability using Grad-CAM
- Generalization behavior across heterogeneous chest X-ray datasets

The study evaluates four experimental settings:

| Training Dataset | Testing Dataset | Evaluation Type |
|---|---|---|
| Kaggle | Kaggle | In-Domain |
| Kaggle | RSNA | Cross-Domain |
| RSNA | RSNA | In-Domain |
| RSNA | Kaggle | Cross-Domain |

---

## Key Features

- Comparative analysis of VGG16, ResNet50, and DenseNet121
- Cross-dataset evaluation framework for medical imaging AI
- Grad-CAM based interpretability analysis
- Transfer learning with pretrained CNN architectures
- Quantitative robustness evaluation under domain shift
- Performance analysis using Accuracy, Precision, Recall, F1-Score, and AUC

---

## Datasets

### Kaggle Chest X-ray Dataset
Public pneumonia classification dataset containing normal and pneumonia chest X-ray images.

### RSNA Pneumonia Detection Dataset
Large-scale radiology dataset designed for pneumonia localization and detection tasks.

---

## Model Architectures

| Model | Key Characteristics |
|---|---|
| VGG16 | Sequential CNN architecture with deep convolutional layers |
| ResNet50 | Residual learning with skip connections |
| DenseNet121 | Dense feature connectivity and improved gradient flow |

---

## Methodology

### Data Preprocessing

- Image resizing to 224×224
- RGB conversion
- Pixel normalization
- Data augmentation using:
  - Horizontal flipping
  - Rotation
  - Zoom transformations

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 10–15 |
| Loss Function | Binary Cross-Entropy |

---

## Experimental Results

### In-Domain Performance

| Model | Dataset | Accuracy | AUC |
|---|---|---|---|
| VGG16 | Kaggle | 81% | 89% |
| DenseNet121 | Kaggle | 87% | 92% |
| DenseNet121 | RSNA | 73% | 81% |

### Cross-Domain Performance

| Training → Testing | Model | Accuracy | AUC |
|---|---|---|---|
| Kaggle → RSNA | DenseNet121 | 45% | 70% |
| RSNA → Kaggle | DenseNet121 | 76% | 85% |
| Kaggle → RSNA | ResNet50 | 54% | 63% |

### Key Findings

- DenseNet121 demonstrated the strongest cross-domain robustness
- Significant performance degradation occurred under domain shift
- Cross-dataset validation is essential for clinically deployable AI systems
- Grad-CAM analysis revealed attention instability under domain transfer

---

## Grad-CAM Interpretability

Grad-CAM visualizations were used to analyze model attention behavior during in-domain and cross-domain evaluations.

Key observations:

- In-domain models focused on clinically relevant lung regions
- Cross-domain predictions occasionally highlighted irrelevant regions
- DenseNet121 maintained more stable attention localization

---

## Project Structure

```bash
domain-shift-pneumonia-detection/
│
├── Final_IEEE_Paper.pdf
├── README.md
│
├── notebooks/
│   ├── vgg16_model.ipynb
│   ├── resnet50_model.ipynb
│   └── densenet121_model.ipynb
│
├── results/
│   ├── confusion_matrices/
│   ├── roc_curves/
│   └── gradcam_outputs/
│
└── images/
```

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## Research Contributions

This work contributes to medical imaging AI research by:

- Evaluating cross-dataset robustness systematically
- Comparing architectural stability under domain transfer
- Combining quantitative evaluation with interpretability analysis
- Highlighting limitations of single-dataset benchmarking

---

## Limitations

- Evaluation limited to two datasets
- No domain adaptation techniques applied
- Computational constraints restricted larger-scale experiments
- Grad-CAM analysis remains partially qualitative

---

## Future Work

- Domain adaptation using adversarial learning
- Multi-institutional evaluation datasets
- Attention-based hybrid architectures
- Quantitative attention consistency metrics
- Clinical metadata integration

---

## Research Paper

The complete research paper is included in this repository:

`Final_IEEE_Paper.pdf`

---

## Citation

```bibtex
@article{domainshift2026,
  title={Investigating Domain Shift in Deep CNN Models for Pneumonia Detection Across Kaggle and RSNA Chest X-Ray Datasets},
  author={Siddique, Mohammad Aadil and Siddique, Mohammad Yasin and others},
  year={2026}
}
```

---

## License

This repository is intended for academic and research purposes.
