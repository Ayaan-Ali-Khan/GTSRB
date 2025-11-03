# 🚦 Traffic Sign Classification Web App

A Streamlit-based web application that classifies German Traffic Signs (GTSRB dataset) using a deep learning model. Users can upload an image of a traffic sign, and the app predicts its class in real-time.  
This project showcases end-to-end deployment of a machine learning model — from training to a fully interactive web interface.

<img src="download.png">
---

## 📂 Project Overview

- 🧠 **Deep Learning Model** trained on the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset  
- ✅ Achieved around **97% test accuracy**  
- 🌐 **Deployed using Streamlit Cloud**  
- 📁 Includes model, app interface, and prediction pipeline

---

## 🧠 Model Details

| Feature            | Description                                   |
|--------------------|-----------------------------------------------|
| Architecture       | Convolutional Neural Network (CNN)           |
| Dataset            | GTSRB (43 traffic sign classes)              |
| Accuracy           | ~97% on test data                            |
| Framework          | TensorFlow / Keras                           |

---

## 📊 App Features

✔ Upload a traffic sign image  
✔ Real-time prediction with confidence score  
✔ Sidebar navigation: **Home**, **About Model**, **About Data**  
✔ Clean UI with custom styling  
✔ Deployed and accessible online

---

## 🚀 Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ayaan-Ali-Khan/GTSRB.git
cd GTSRB
```

### 2️ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```