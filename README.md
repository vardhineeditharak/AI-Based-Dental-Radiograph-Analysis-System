# 🦷 AI-Based Dental Radiograph Analysis System

---

# 🌟 Introduction

The **AI-Based Dental Radiograph Analysis System** is a computer vision and machine learning project designed to analyze **panoramic dental X-ray images (OPG)** and assist dentists in identifying dental conditions automatically.

Dental radiographs are widely used to detect cavities, implants, impacted teeth, and other dental abnormalities. However, manual analysis of these radiographs can be time-consuming and requires expertise.

This project explores the use of **Artificial Intelligence, Computer Vision, and Machine Learning** to automate dental radiograph analysis. The system processes dental X-ray images, segments tooth regions, extracts visual features, and predicts dental conditions using a trained machine learning model.

The goal is to build a **foundation for AI-assisted diagnostic systems in digital dentistry**.

---

# ✨ Features

* Automated dental X-ray image analysis
* Tooth region detection using image processing techniques
* Feature extraction from dental radiographs
* Machine learning model for dental condition classification
* Visualization of predicted results on dental X-rays
* Modular pipeline for training and prediction

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

# 🛠️ Tech Stack

### Programming Language

* Python

### Libraries

* OpenCV – image processing and segmentation
* NumPy – numerical computations
* Pandas – dataset handling
* Scikit-learn – machine learning model training
* Matplotlib – visualization
* TensorFlow – deep learning framework support

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

# Dataset

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

Create a virtual environment (recommended)

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
Performs tooth detection and segmentation.

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

# 🔧 Development

To contribute or modify the project:

1. Fork the repository

2. Create a new branch

```bash
git checkout -b feature-name
```

3. Make changes and commit

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Create a pull request

---

# Future Improvements

Possible improvements include:

* Deep learning based tooth segmentation using **U-Net**
* Tooth detection using **YOLO**
* Automatic **FDI tooth numbering system**
* Detection of additional dental diseases
* Integration with a web-based diagnostic dashboard

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

# License

This project is intended for **educational and research purposes**.

---

If you want, I can also **upgrade this README one more level** by adding:

* GitHub **badges (Python, TensorFlow, License)**
* **Architecture diagram image**
* **Sample dental detection output images**

That makes the repo look **like a professional AI project rather than a normal student repo**.
