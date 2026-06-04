# Wearable Sensor Activity Recognition : Deep Learning (STAT41120)

## Overview
Comparison of two deep learning architectures for 19-class daily and sports activity recognition using wearable sensor data.

## Dataset
125 time steps × 45 sensor channels across 5 body-mounted units.  
- Training: 8,170 observations | Test: 950 observations  
- Dataset not included in this repo (UCD course material).

## Models
| Model | Architecture | Test Accuracy |
|-------|-------------|---------------|
| DNN   | 512 → 256 → 128 (BatchNorm + Dropout) | ~98.2% |
| 1D CNN | 3× Conv blocks (64→128→256 filters) + GAP | **99.16%** |

## Key Findings
- CNN outperformed DNN significantly, leveraging temporal structure in time series
- All 8 CNN misclassifications involved elevator activities (up/down confusion)
- Hyperparameter tuning via grid search over learning rate, batch size, and dropout

## Files
- `activity_recognition.R` : Full pipeline: preprocessing, model building, tuning, evaluation
- `report.docx` : Written report with architecture diagrams, comparison tables, and confusion matrix

## Requirements
R with `keras3`, `tensorflow`, `reticulate`
