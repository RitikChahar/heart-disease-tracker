# Heart Disease Prediction Using Machine Learning

An academic internship project that builds a binary classification model to predict whether a patient has heart disease, using the UCI Heart Disease (Cleveland) dataset. The project includes a Jupyter Notebook for analysis, a training script to export the best model, and an interactive Streamlit web dashboard for live predictions.

---

## Project Structure

```
Heart-Disease-Prediction/
├── heart.csv                          ← UCI Heart Disease dataset (303 records)
├── heart_disease_prediction.ipynb     ← Main Jupyter Notebook (EDA + model training)
├── train_and_save_model.py            ← Script to retrain and export the best model
├── app.py                             ← Streamlit web dashboard (live predictions)
├── generate_dashboard_screenshot.py   ← Utility to capture a dashboard screenshot
├── requirements.txt                   ← Python dependencies
├── models/
│   ├── heart_disease_model.pkl        ← Serialised scikit-learn Pipeline (auto-generated)
│   ├── dashboard_screenshot.png       ← Screenshot of the Streamlit app
│   └── README.md                      ← Notes on the saved model format
└── README.md                          ← This file
```

---

## Dataset

**UCI Heart Disease — Cleveland subset**  
303 patients · 13 input features · 1 binary target (`0` = no disease, `1` = disease present)

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1/0) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1/0) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment (0–2) |
| `ca` | Number of major vessels coloured by fluoroscopy (0–3) |
| `thal` | Thalassemia (1–3) |

---

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
streamlit
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Run

### 1 — Jupyter Notebook (EDA & Model Analysis)

```bash
jupyter notebook heart_disease_prediction.ipynb
```

Run all cells from top to bottom (`Cell → Run All`).

---

### 2 — Train & Export the Model

Run this once to train all four models and save the best one (by F1-Score) to `models/heart_disease_model.pkl`:

```bash
python train_and_save_model.py
```

The script performs an **80/20 stratified split** with `random_state=42`, so results are fully reproducible.

---

### 3 — Launch the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard loads `models/heart_disease_model.pkl` and lets you enter patient data interactively to get an instant prediction with a probability score.

> **Note:** Run `train_and_save_model.py` first if `heart_disease_model.pkl` does not exist.

---

## Notebook Sections

| # | Section |
|---|---------|
| 1 | Imports & Setup |
| 2 | Data Loading |
| 3 | Data Exploration |
| 4 | Missing Value Analysis |
| 5 | Exploratory Data Analysis (EDA) |
| 6 | Data Preprocessing & Train/Test Split |
| 7 | Model Training |
| 8 | Model Evaluation |
| 9 | Best Model Selection |
| 10 | Prediction Function |
| 11 | Sample Predictions |

---

## Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | `max_iter=1000` |
| K-Nearest Neighbours (KNN) | `n_neighbors=5` |
| Decision Tree | `max_depth=5` |
| Random Forest | `n_estimators=100` |

Each model is wrapped in a **scikit-learn `Pipeline`** (StandardScaler → Classifier). The model with the highest **F1-Score** on the test set is saved.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Streamlit Dashboard

The [`app.py`](app.py) dashboard provides:

- A **3-column input form** for all 13 clinical features
- Input validation with user-friendly error messages
- A **Predict** button that returns:
  - Binary prediction (Heart Disease / No Heart Disease)
  - Probability of heart disease (with a progress bar)
  - The name of the model used
- A sidebar **Feature Reference** table
- An **educational disclaimer** (not for clinical use)

---

## Quick Prediction Example (Notebook)

```python
predict_heart_disease({
    'age': 55, 'sex': 1, 'cp': 0, 'trestbps': 140,
    'chol': 250, 'fbs': 1, 'restecg': 1, 'thalach': 150,
    'exang': 1, 'oldpeak': 2.0, 'slope': 1, 'ca': 1, 'thal': 2
})
```

---

## Disclaimer

This project is built for **academic and educational purposes only**.  
The predictions generated are **not a medical diagnosis** and must not be used as a substitute for professional medical advice, examination, or treatment.  
Always consult a qualified healthcare professional for any health concerns.
