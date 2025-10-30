# Intelligent Dental Handpiece: Real-Time Motion Analysis for Skill Development

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Fs25206489-blue)](https://doi.org/10.3390/s25206489)
[![Journal](https://img.shields.io/badge/Journal-Sensors%20(Q1)-green)](https://www.mdpi.com/1424-8220/25/20/6489)
[![License](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset](https://img.shields.io/badge/Dataset-Mendeley%20Data-orange)](https://doi.org/10.17632/h76rf38jkn.1)

## 📋 Overview

The **Intelligent Dental Handpiece (IDH)** is a next-generation training tool that combines motion sensors and machine learning to provide real-time feedback during dental procedures. This system helps dental students and professionals improve their dexterity, precision, and technique through objective motion analysis.

## 🔬 Research Publication

This repository contains the implementation and dataset for our research published in **Sensors (Q1 Journal)**:

**Title:** Intelligent Dental Handpiece: Real-Time Motion Analysis for Skill Development

**Authors:** Mohamed Sallam, Yousef Salah, Yousef Osman, Ali Hegazy, Esraa Khatab, Omar Shalash

**Published:** October 21, 2025

**Read the full paper:** [https://www.mdpi.com/1424-8220/25/20/6489](https://www.mdpi.com/1424-8220/25/20/6489)

**Citation:**
```bibtex
@article{sallam2025intelligent,
  title={Intelligent Dental Handpiece: Real-Time Motion Analysis for Skill Development},
  author={Sallam, Mohamed and Salah, Yousef and Osman, Yousef and Hegazy, Ali and Khatab, Esraa and Shalash, Omar},
  journal={Sensors},
  volume={25},
  number={20},
  pages={6489},
  year={2025},
  publisher={MDPI},
  doi={10.3390/s25206489}
}
```

## ✨ Key Features

- **Real-time motion tracking** using IMU sensors (LSM6DS3TR-C)
- **Three motion state classification:**
  - ✅ Lever Range (0°-10° deviation)
  - ⚠️ Alert (10°-15° deviation)
  - 🛑 Stop Range (>15° deviation)
- **Visual and auditory feedback** via OLED display and buzzer
- **Cloud-based analytics** with Firebase integration
- **Machine Learning models** achieving up to 100% accuracy

## 🎯 Performance Results

| Model | Test Accuracy | Precision | Recall | F1-Score |
|-------|--------------|-----------|---------|----------|
| **Logistic Regression** | **100%** | 1.000 | 1.000 | 1.000 |
| **Random Forest** | **100%** | 1.000 | 1.000 | 1.000 |
| Linear SVM | 99.6% | 0.996 | 0.996 | 0.996 |
| SVM Polynomial | 99.6% | 0.996 | 0.996 | 0.996 |
| SVM RBF | 99.19% | 0.992 | 0.992 | 0.992 |
| Neural Network | 98.52% | 0.985 | 0.985 | 0.985 |

All models achieved **perfect ROC AUC = 1.00** across all classes.

## 🔧 Hardware Components

- **Microcontroller:** XIAO nRF52840
- **IMU Sensor:** LSM6DS3TR-C (6-DOF motion tracking)
- **Display:** 0.96" OLED SSD1306 (128×64)
- **Feedback:** Active buzzer for audio alerts
- **Handpiece:** NSK Air-turbine drill
- **Communication:** Wi-Fi enabled for cloud data upload

## 📊 Dataset

The dataset contains **3,720 records** from **61 practitioners** (29 male, 32 female) with an average age of 38 years.

**Dataset Features:**
- Roll, pitch, yaw (orientation)
- Acceleration and velocity
- Deviation angle
- Time stamps
- Device type
- Motion state labels

**Download Dataset:** [Mendeley Data](https://doi.org/10.17632/h76rf38jkn.1)

**Class Distribution:**
- Class 0 (Alert): 1,239 samples (33.3%)
- Class 1 (Lever Range): 1,265 samples (34.0%)
- Class 2 (Stop Range): 1,216 samples (32.7%)

## 🚀 Installation

### Prerequisites

```bash
Python 3.9+
pip install -r requirements.txt
```

### Required Libraries

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.2.0
tensorflow>=2.11.0
matplotlib>=3.5.0
seaborn>=0.11.0
shap>=0.41.0
firebase-admin>=6.0.0
```

### Hardware Setup

1. Connect LSM6DS3TR-C IMU to XIAO nRF52840 via I2C
2. Connect OLED display to designated GPIO pins
3. Connect buzzer for audio feedback
4. Upload firmware to microcontroller
5. Configure Firebase credentials for cloud sync

## 💻 Usage

### Training Models

```python
from src.train import train_models
from src.data_loader import load_dataset

# Load preprocessed dataset
X_train, X_test, y_train, y_test = load_dataset('data/dental_motion_data.csv')

# Train all models
results = train_models(X_train, y_train, X_test, y_test)

# View performance metrics
print(results['logistic_regression'])
print(results['random_forest'])
```

### Real-Time Classification

```python
from src.realtime import RealtimeClassifier

# Initialize classifier with trained model
classifier = RealtimeClassifier(model_path='models/logistic_regression.pkl')

# Start real-time monitoring
classifier.start_monitoring(device_port='/dev/ttyUSB0')
```

### Visualization

```python
from src.visualization import plot_confusion_matrix, plot_roc_curves

# Generate confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=['Alert', 'Lever', 'Stop'])

# Plot ROC curves
plot_roc_curves(y_test, y_pred_proba, n_classes=3)
```

## 📁 Repository Structure

```
intelligent-dental-handpiece/
├── data/
│   ├── raw/                    # Raw IMU sensor data
│   ├── processed/              # Preprocessed datasets
│   └── README.md              # Dataset documentation
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── svm_linear.pkl
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Feature engineering
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Performance evaluation
│   ├── realtime.py           # Real-time classification
│   └── visualization.py      # Plotting functions
├── firmware/
│   └── arduino_sketch.ino    # Microcontroller code
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── model_comparison.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## 🔍 Key Findings

1. **Linear separability validated:** Simple linear models (Logistic Regression, Linear SVM) matched or outperformed complex non-linear models
2. **Feature importance:** Deviation and Take Time were the most influential predictors across all models
3. **Perfect classification:** Both Logistic Regression and Random Forest achieved 100% accuracy with excellent calibration
4. **Computational efficiency:** Linear models require <1 second training time with sub-millisecond inference

## 🎓 Applications

- **Dental Education:** Objective skill assessment for students
- **Professional Training:** Continuous improvement tracking
- **Quality Assurance:** Standardized procedure monitoring
- **Research:** Motion analysis and ergonomics studies

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)**.

See [LICENSE](LICENSE) file for details.

## 📧 Contact

**Mohamed Sallam**
- Email: mohsallam63@gmail.com
- LinkedIn: [mohamed-sallam](https://www.linkedin.com/in/mohamed-sallam-3312a91ba)
- GitHub: [mohsallam1](https://github.com/mohsallam1)

**Corresponding Author: Dr. Omar Shalash**
- Email: o.shalash@ajman.ac.ae
- Institution: Ajman University, UAE

## 🙏 Acknowledgments

- Arab Academy for Science, Technology and Maritime Transport (AASTMT)
- Ajman University
- Heriot-Watt University
- All 61 practitioners who participated in data collection

## 📚 Related Publications

- [A Multimodal Polygraph Framework with Optimized Machine Learning for Robust Deception Detection](https://www.mdpi.com/2411-5134/10/6/96) - IoT Journal (Q1), MDPI

## 🔗 Useful Links

- **Paper:** [https://www.mdpi.com/1424-8220/25/20/6489](https://www.mdpi.com/1424-8220/25/20/6489)
- **Dataset:** [https://doi.org/10.17632/h76rf38jkn.1](https://doi.org/10.17632/h76rf38jkn.1)
- **MDPI Sensors Journal:** [https://www.mdpi.com/journal/sensors](https://www.mdpi.com/journal/sensors)

---

**⭐ If you find this work useful, please consider citing our paper and starring this repository!**

**Status:** ✅ Published in Sensors (Q1) | 📊 Dataset Available | 🔓 Open Source
