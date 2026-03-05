# 🦷 AI-Based Dental Radiograph Analysis System

---

## 🎥 Project Demo

[![Watch the Demo](<img width="1470" height="956" alt="Screenshot 2026-03-05 at 10 59 11 PM" src="https://github.com/user-attachments/assets/b2aeb1c0-78a6-46a4-9664-5c7660c158bb" />)](https://drive.google.com/file/d/1vRvN1LPiEssgfCQoisZSsPkKMxlRj8pN/view?usp=sharing)

# 🌟 Introduction

The **AI-Based Dental Radiograph Analysis System** is a computer vision and machine learning project designed to analyze **panoramic dental X-ray images (OPG)** and assist dentists in identifying dental conditions automatically.

Dental radiographs are an essential diagnostic tool used in dentistry to detect cavities, implants, impacted teeth, bone loss, and other abnormalities. However, analyzing these radiographs manually can be time-consuming and requires trained professionals.

This project explores the use of **Artificial Intelligence (AI), Computer Vision, and Machine Learning** to automate dental radiograph analysis. The system processes dental X-ray images, segments tooth regions, extracts visual features, and predicts dental conditions using a trained machine learning model.

The goal of this project is to demonstrate how **AI-assisted systems can support dentists in faster and more accurate diagnostic decision-making**, paving the way for future smart dental healthcare systems.

---

# 🎯 Objectives

The primary objectives of this project include:

* Automating the analysis of panoramic dental radiographs
* Detecting tooth regions using computer vision techniques
* Extracting useful features from dental X-ray images
* Classifying dental conditions using machine learning
* Providing visual outputs to assist dental professionals

---

# ✨ Features

* Automated dental X-ray image analysis
* Tooth region detection using image processing techniques
* Feature extraction from dental radiographs
* Machine learning model for dental condition classification
* Visualization of predicted results on dental X-rays
* Modular pipeline for training and prediction
* Easy dataset integration for future training
* Extendable architecture for deep learning models

---

# 🏗️ Architecture

The system follows a **computer vision pipeline** for analyzing dental radiographs.

```
Dental X-ray Image
        ↓
Image Preprocessing
        ↓
Tooth Segmentation
        ↓
Feature Extraction
        ↓
Machine Learning Model
        ↓
Dental Condition Classification
        ↓
Visualization of Results
```

---

# ⚙️ Methodology

The system follows a multi-step methodology:

### 1. Image Acquisition

Dental radiograph images are provided as input to the system.

### 2. Image Preprocessing

The images are processed to enhance quality and remove noise using:

* grayscale conversion
* filtering
* contrast adjustment

### 3. Tooth Segmentation

Image segmentation techniques are used to isolate teeth from the surrounding structures.

This is achieved using:

* thresholding
* contour detection
* region extraction

### 4. Feature Extraction

After segmentation, important features are extracted such as:

* contour area
* intensity distribution
* texture patterns
* edge features

These features help the model identify different dental conditions.

### 5. Machine Learning Classification

A machine learning model is trained on labeled features to classify dental conditions.

### 6. Visualization

Detected regions and predicted labels are displayed on the dental X-ray image for interpretation.

---

# 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* **OpenCV** – image processing and segmentation
* **NumPy** – numerical computations
* **Pandas** – dataset handling
* **Scikit-learn** – machine learning model training
* **Matplotlib** – visualization
* **TensorFlow** – deep learning framework support

### Tools

* Jupyter Notebook
* Google Colab
* GitHub

---

# 📋 Prerequisites

Before running the project, ensure the following are installed:

* **Python 3.10**

This project should be run using **Python 3.10** because some dependencies, especially **TensorFlow**, may not work properly with newer Python versions.

Other requirements include:

* pip
* virtual environment (recommended)

---

# 📊 Dataset

The dataset contains dental radiograph images used for training and testing the machine learning model.

Expected dataset structure:

```
unlabeled_images/
    image1.jpg
    image2.jpg
    image3.jpg

_annotations.csv
```

The **_annotations.csv** file contains labels for the images that are used during training.

Typical labels may include dental conditions such as:

* Normal tooth
* Filling
* Implant
* Impacted tooth

A larger dataset improves the model’s accuracy and generalization.

---

# 🚀 Setup & Installation

Clone the repository

```bash
git clone https://github.com/vardhineeditharak/AI-Based-Dental-Radiograph-Analysis-System.git
```

Navigate to the project folder

```bash
cd AI-Based-Dental-Radiograph-Analysis-System
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

### Windows

```bash
venv\Scripts\activate
```

### Mac / Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🏃‍♂️ Running the Application

### Train the model

```bash
python train.py
```

This script trains the machine learning model using the annotated dataset.

---

### Perform tooth segmentation

```bash
python tooth_segmentation.py
```

This script detects tooth regions from dental X-ray images.

---

### Run dental analysis

```bash
python analyzer.py
```

The analyzer loads the trained model and predicts dental conditions.

---

# 📁 Project Structure

```
AI-Based-Dental-Radiograph-Analysis-System
│
├── analyzer.py
├── train.py
├── tooth_segmentation.py
├── _annotations.csv
├── requirements.txt
│
├── unlabeled_images/
│
└── README.md
```

### File Descriptions

**train.py**

Used to train the machine learning model.

**tooth_segmentation.py**

Performs tooth detection and segmentation using computer vision.

**analyzer.py**

Main analysis script used for prediction.

**_annotations.csv**

Dataset labels used during training.

**requirements.txt**

Contains required Python dependencies.

---

# ⚙️ Environment Configuration

Recommended environment configuration:

**Python version:** `3.10`

Create virtual environment

```bash
python -m venv venv
```

Activate environment

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

This ensures that all dependencies including **TensorFlow** install correctly.

---

# 🔬 Model Explanation

The system uses a **Random Forest Machine Learning Model** for classification.

Random Forest is chosen because:

* It performs well with structured feature datasets
* It handles nonlinear relationships effectively
* It is less prone to overfitting compared to single decision trees

The model is trained using labeled features extracted from dental radiographs.

---

# 📈 Applications

This project has applications in several areas of dentistry:

* Digital dental diagnosis
* Clinical decision support systems
* Dental radiograph interpretation
* Dental education and research
* AI-assisted healthcare systems

---

# ⚠️ Limitations

Some limitations of the current system include:

* Limited dataset size
* Basic segmentation approach
* Limited dental condition categories
* No automatic tooth numbering system

Future versions can improve these aspects using deep learning.

---

# 🔮 Future Improvements

Possible improvements include:

* Deep learning based tooth segmentation using **U-Net**
* Tooth detection using **YOLO**
* Automatic **FDI tooth numbering system**
* Detection of additional dental diseases
* Integration with a web-based diagnostic dashboard
* Cloud-based AI dental analysis platform

---

# References

OpenCV
[https://opencv.org/](https://opencv.org/)

Scikit-learn
[https://scikit-learn.org/](https://scikit-learn.org/)

TensorFlow
[https://www.tensorflow.org/](https://www.tensorflow.org/)

Reference Repository
[https://github.com/adityanandanx/dental-conditions-detection](https://github.com/adityanandanx/dental-conditions-detection)

---

# 🙏 Acknowledgements

This project was inspired by research in **AI-based medical image analysis and dental radiograph interpretation**.

We also acknowledge open-source libraries and repositories that support machine learning research.

---

# License

This project is intended for **educational and research purposes**.

---
