# Retinal Disease Classification Using Transfer Learning

## Overview

Early detection of retinal diseases is crucial for preventing vision impairment and blindness. This project benchmarks six state-of-the-art convolutional neural network (CNN) models on a publicly available ocular fundus image dataset to classify eight retinal disease categories. We explore transfer learning with both feature extraction and full fine-tuning strategies, apply image preprocessing and augmentation, and analyze model interpretability with Grad-CAM.

---

## Dataset

- **Source:** Ocular Disease Intelligent Recognition (ODIR) dataset from Kaggle, containing 6,392 color fundus images from 5,000 patients.
- **Classes:** Eight diagnostic categories including Normal, Diabetes, Glaucoma, Cataract, Age-related Macular Degeneration, Hypertension, Pathological Myopia, and Other abnormalities.
- **Splitting:** Patient-wise stratified split into training (70%), validation (20%), and test (10%) sets to prevent data leakage.
- **Preprocessing:** Applied CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast and highlight subtle retinal features. Slight rotational augmentation (±3°) was used on training images only.

---

## Methods

### Model Architectures

- Six CNNs were evaluated:
  - DenseNet121
  - VGG19
  - MobileNetV3
  - XceptionNet
  - ResNet50
  - InceptionV3

### Transfer Learning Strategies

- **Feature Extraction:** Freeze pretrained backbones and only train classification heads (DenseNet121, VGG19, MobileNetV3, XceptionNet).
- **Full Fine-tuning:** Fine-tune all layers including backbone (ResNet50, InceptionV3).

### Training Setup

- Initialized with ImageNet weights.
- Modified classification heads for 8 classes.
- Grid search over learning rates and weight decays.
- Used Adam optimizer, early stopping, max 10 epochs.
- Best model selected based on validation accuracy.

### Evaluation Metrics

- Accuracy
- Weighted Precision, Recall, F1-score (to handle class imbalance)
- Focus on F1-score for balanced performance assessment

---

## Results

- **Best Model:** InceptionV3 (full fine-tuning) with F1-score of 57.5%
- DenseNet121 and MobileNetV3 performed well under feature extraction settings
- ResNet50 and XceptionNet had relatively lower performance
- Grad-CAM visualizations showed interpretable patterns in most classes, highlighting key anatomical regions
- Confusion matrix revealed strengths in detecting cataract and pathological myopia but challenges with age-related macular degeneration and diabetes

---

## Conclusions

- Transfer learning with full fine-tuning yields superior classification results on retinal fundus images
- Explainability methods like Grad-CAM enhance trust and clinical relevance by highlighting model decision regions
- Future work includes creating hybrid and lightweight models to improve accuracy and computational efficiency for deployment in low-resource settings

---

## Usage

Refer to the accompanying Jupyter notebooks for:

- Data preprocessing and augmentation
- Model training and evaluation
- Generating Grad-CAM visualizations

---

## Libraries Used

- **Python Core**
  - os
  - zipfile
  - time
  - copy
  - random

- **Numerical & Data Handling**
  - numpy
  - pandas

- **Visualization**
  - matplotlib
  - seaborn

- **Machine Learning & Deep Learning**
  - sklearn (scikit-learn)
  - torch (PyTorch)
  - torchvision
  - timm (PyTorch image models)
  - tensorflow
  - tensorflow.keras (layers, models, regularizers, optimizers, callbacks)

- **Utilities**
  - tqdm
  - PIL (Python Imaging Library)

---

## References

1. Ocular Disease Intelligent Recognition (ODIR) Dataset: https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k
2. Jameel, D., & Abdulazeez, A. M. (2024). Ocular Disease Recognition Based on Deep Learning: A Comprehensive Review. The Indonesian Journal of Computer Science, 13(3).
3. Herrera-Chavez, A. I., Rodríguez-Martínez, E. A., Flores-Fuentes, W., Rodgíruez-Quiñonez, J. C., García-Gallegos, J. C., Montiel-Ross, O. H., ... & Sergiyenko, O. (2024, June). Multi-label Image Classification for Ocular Disease Diagnosis Using K-fold Cross-Validation on the ODIR-5K Dataset. In 2024 IEEE 33rd International Symposium on Industrial Electronics (ISIE) (pp. 1-6). IEEE
4. Salehi, A. W., Khan, S., Gupta, G., Alabduallah, B. I., Almjally, A., Alsolai, H., ... & Mellit, A. (2023). A study of CNN and transfer learning in medical imaging: Advantages, challenges, future scope. Sustainability, 15(7), 5930.
5. Kaur, R., Kumar, R., & Gupta, M. (2021, December). Review on transfer learning for convolutional neural network. In 2021 3rd International Conference on Advances in Computing, Communication Control and Networking (ICAC3N) (pp. 922-926). IEEE.
6. Wanto, A., Yuhandri, Y., & Okfalisa, O. (2023, August). Optimization Accuracy of CNN Model by Utilizing CLAHE Parameters in Image Classification Problems. In 2023 International Conference on Networking, Electrical Engineering, Computer Science, and Technology (IConNECT) (pp. 253-258). IEEE.

---

## Acknowledgments

The project is a part of the final course requirement for Machine Learning for Medical Applications at Johns Hopkins University. Thanks to the original dataset providers and my team members Akshay Bhuvaneshwari Ramakrishnan, Alex Zhu, Jianzhi Shen and Liora Dsilva for their contrubution to the project.

---

