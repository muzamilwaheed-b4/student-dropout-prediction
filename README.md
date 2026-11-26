# 🎓 Student Dropout & Academic Success Prediction

A machine learning web application to predict university student outcomes:  
**Dropout** | **Enrolled** | **Graduate**

> **Base Reference:** [AmaedaQ/Student-Dropout-Prediction-System](https://github.com/AmaedaQ/Student-Dropout-Prediction-System)  
> **Our improvements:** 3-class prediction, SMOTE, K-Means clusters, feature importance, risk badges, suggestions

---

## 📁 Project Structure

```
student-dropout-prediction/
├── data/
│   └── data.csv          ← Download from Kaggle (see below)
├── model/
│   └── model_bundle.pkl  ← Generated after running train.py
├── static/
│   └── plots/            ← EDA + model charts (auto-generated)
├── templates/
│   ├── index.html
│   ├── prediction.html
│   ├── results.html
│   └── about.html
├── train.py              ← Run this FIRST
├── app.py                ← Flask web app
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download from Kaggle: [Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)

Place the CSV file as: `data/data.csv`

> Dataset has `;` as separator — the script handles it automatically.

### 3. Train the model
```bash
python train.py
```
This will:
- Load and preprocess data (missing values, outliers, encoding)
- Run EDA and save plots to `static/plots/`
- Train 4 models WITH and WITHOUT SMOTE
- Run K-Means clustering (3 risk groups)
- Select best model by Macro F1
- Save model to `model/model_bundle.pkl`

### 4. Run the web app
```bash
python app.py
```
Open: http://127.0.0.1:5000

---

## 🧪 Lab Concepts Applied

| Lab | Concept | Where Used |
|-----|---------|------------|
| Lab 07 | Label & Ordinal Encoding | Target encoding in `train.py` |
| Lab 08 | Outlier Handling (IQR) | Grade column capping |
| Lab 09 | Feature Scaling | StandardScaler for KNN + LR |
| Lab 10 | Imbalance Handling | SMOTE (only on training data) |
| Lab 11 | K-Means Clustering | At-risk student groups |
| Lab 12 | Evaluation Metrics | Accuracy, Macro F1, Confusion Matrix, CV |

---

## 📊 Models

| Model | Why Used |
|-------|----------|
| KNN (k=7) | Distance-based, simple baseline |
| Decision Tree (depth=8) | Interpretable rules |
| **Random Forest (200 trees)** | Best performance, feature importance |
| Logistic Regression | Linear baseline, fast |

**Best model selected by Macro F1** (not accuracy) because classes are imbalanced.

---

## 🌟 Our Improvements Over Base Repo

1. ✅ **3-class prediction** (not binary Dropout/Not-Dropout)
2. ✅ **SMOTE applied correctly** (only on train data — no data leakage)
3. ✅ **K-Means risk clusters** (High/Medium/Low risk groups)
4. ✅ **Feature importance chart** in app
5. ✅ **Risk % with probability bar chart** in prediction page
6. ✅ **Personalized suggestions** based on prediction
7. ✅ **SMOTE vs No-SMOTE comparison** plot
8. ✅ **5-fold cross-validation** for reliability

---

## 📈 Dataset

**UCI "Predict Students' Dropout and Academic Success"**  
Realinho, V., et al. (2021). Portuguese university — 4,424 students, 36 features.

- Dropout: 32%
- Enrolled: 17%  
- Graduate: 49%

---

## 🛠️ Technologies

- **Python** — pandas, numpy, scikit-learn, imbalanced-learn
- **Flask** — web framework
- **Bootstrap 5** — frontend UI
- **Chart.js** — probability bar chart
- **Matplotlib / Seaborn** — EDA plots
