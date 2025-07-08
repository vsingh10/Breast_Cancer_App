# 🧬 Breast Cancer Predictor

[![Streamlit App](https://img.shields.io/badge/Launch%20App-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://breastcancerpredictapp.streamlit.app/)  
A machine learning-powered web app that predicts whether a breast tumor is **Benign** or **Malignant** using clinical data from digitized breast biopsies.

---
## 📁 Folder Structure
```bash
BREAST_CANCER_APP/
├── app/
│   └── main.py
├── assets/
├── data/
│   └── data.csv
├── model/
│   ├── main.py
│   ├── model.pkl
│   └── scalar.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Overview

The model uses the **Breast Cancer Wisconsin (Diagnostic) Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)). It contains features calculated from digitized images of a fine needle aspirate (FNA) of a breast mass.

| id       | diagnosis | radius_mean | texture_mean | perimeter_mean | area_mean | ... | fractal_dimension_worst |
|----------|-----------|-------------|--------------|----------------|-----------|-----|--------------------------|
| 842302   | M         | 17.99       | 10.38        | 122.8          | 1001.0    | ... | 0.1189                   |
| 842517   | M         | 20.57       | 17.77        | 132.9          | 1326.0    | ... | 0.08902                  |
| ...      | ...       | ...         | ...          | ...            | ...       | ... | ...                      |

- **Total Samples**: 569  
- **Classes**:  
  - `M` = Malignant  
  - `B` = Benign  
- **Features**: 30 numeric values per sample

---

## 🧠 Model Information

- **Algorithm**: Logistic Regression  
- **Scaler**: StandardScaler (scikit-learn)  
- **Accuracy**: ~97% on test data  
- **Target Variable**: `diagnosis`  
- **Output**: Binary classification → Benign (0) or Malignant (1)

---

## 🚀 Key Features

- 🎚️ Adjustable input sliders for 30 features  
- 📊 Interactive **Radar Chart** to visualize input features  
- ⚡ Real-time predictions with probabilities  
- 🧾 Color-coded and intuitive result display  
- ☁️ Hosted using Streamlit Cloud

---

## 🖼️ Sample Result

### 📌 Prediction Output + Radar Chart

![Prediction Result](https://raw.githubusercontent.com/vsingh10/Breast_Cancer_App/main/assets/image.png)

---

## 🛠️ How to Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/BREAST_CANCER_APP.git
cd BREAST_CANCER_APP
```
### 2. Install Dependencies

```bash
 pip install -r requirements.txt
```
### 3. Train the model 

```bash
python model/main.py
```
### 4. Run the Streamlit App

```bash
streamlit run app/main.py
```

## 📦 Requirements
```bash
streamlit
scikit-learn
pandas
numpy
plotly
emoji
```
---

## ⚠️ Disclaimer

> This application is developed solely for **educational and demonstrative purposes**.  
> It is **not intended for clinical or diagnostic use**. Always consult medical professionals for actual diagnoses.

---

## 🤝 Acknowledgments

- [UCI Machine Learning Repository – Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- [Scikit-learn](https://scikit-learn.org/)
- [Streamlit](https://streamlit.io/)
- [Plotly](https://plotly.com/)

---

## 👨‍💻 Author

**Vishal Singh**  
📧 your.email@example.com  
🔗 [LinkedIn](https://www.linkedin.com/in/vishal-singh10/)

---

## ⭐️ Show Your Support

If you like this project, feel free to:

- ⭐️ Star the repo  
- 🛠️ Fork and contribute  
- 📢 Share it with friends interested in ML and healthcare tech

---

## 📝 License

This project is open source and available under the [MIT License](https://choosealicense.com/licenses/mit/).




