# 🧠 Mental Health Text Classification API

An end-to-end **NLP Machine Learning project** that classifies user-written text into mental health categories using **TF-IDF + Logistic Regression**, deployed as a **FastAPI REST API** on **Render**.

🔗 **Live API URL:**  
https://mentalhealthlabel.onrender.com/

---

## 📌 Problem Statement

Mental health-related text data (social media posts, forums, messages) often contains early indicators of emotional distress.  
The goal of this project is to **automatically classify text into mental health categories** to support early detection and analysis.

---

## 🏷️ Target Classes

The model predicts one of the following classes:

- **Normal**
- **Anxiety**
- **Depression**
- **Suicidal**

This is a **multi-class classification problem** with class imbalance.

---

## 🧪 Dataset

- Text-based mental health dataset
- Imbalanced class distribution
- Preprocessed using a **custom text-cleaning pipeline**
- Train/Test split used for evaluation

---

## 🧠 Model & Approach

### 🔹 Text Preprocessing
Custom `TextCleanerTFIDF` transformer applied inside an sklearn pipeline:
- Apostrophe normalization
- Lowercasing
- Contraction expansion
- URL & email removal
- Emoji demojization
- Noise & punctuation cleaning
- Empty-text safety handling

### 🔹 Feature Engineering
- **TF-IDF Vectorization**
  - `ngram_range = (1, 2)`
  - `min_df = 5`
  - `max_features = 20000`
  - `sublinear_tf = True`
  - `norm = "l2"`

### 🔹 Model
- **Logistic Regression**
- `class_weight = "balanced"`
- Hyperparameters tuned using **GridSearchCV**
- Optimized using **macro F1-score**

---

## 📊 Model Performance

### Test Set Results
- Accuracy: ~79%
- Anxiety : ~0.79
- Depression : ~0.70
- Normal : ~0.91
- Suicidal : ~0.70

---

## 🚀 API Endpoints
- Root Link: *https://mentalhealthlabel.onrender.com/*
- Test Link: *https://mentalhealthlabel.onrender.com/docs*
- Endpoint Link (Single Row): *https://mentalhealthlabel.onrender.com/predict*
- Endpoint Link (Multiple Row): *https://mentalhealthlabel.onrender.com/predict-batch*

---
### Single request body (Example)
{
  "text": "I feel very anxious and stressed lately"
}

### Multiple records body (Example)
[
  {"text": "I feel hopeless and tired"},
  {"text": "Life feels good today"},
  {"text": "I am scared and overthinking everything"}
]

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Numpy
- Pandas
- FastAPI
- Uvicorn
- Joblib
- Render (Deployment)

---

## 🌐 Deployment
- Deployed on Render
- Docker-free deployment
- Uses uvicorn as ASGI server
- Model loaded from serialized .pkl file

---
## 🔮 Future Improvements
- Use transformer-based models (BERT)
- Hierarchical classification for better minority-class recall
- Confidence scores in API responses
- Streaming & real-time inference
- Frontend integration

---

## 👤 Author
### Subir Kumar Behera
Aspiring Data Analyst | Machine Learning Enthusiast

