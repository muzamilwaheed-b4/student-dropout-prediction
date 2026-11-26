"""
Student Dropout Prediction - Flask Web App
Based on: AmaedaQ/Student-Dropout-Prediction-System
Improvements:
  - 3-class prediction (Dropout / Enrolled / Graduate)
  - Risk % with probability bar
  - Feature importance chart
  - Personalized suggestions
  - No data leakage (scaler fitted only on train data)
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load saved model bundle ──────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(BASE_DIR, "model", "model_bundle.pkl")

bundle    = None
model     = None
scaler    = None
FEATURES  = []
MODEL_NAME = ""
results_with    = {}
results_without = {}
cv_mean = 0.0
cv_std  = 0.0

def load_bundle():
    global bundle, model, scaler, FEATURES, MODEL_NAME
    global results_with, results_without, cv_mean, cv_std
    if os.path.exists(BUNDLE_PATH):
        bundle          = joblib.load(BUNDLE_PATH)
        model           = bundle["model"]
        scaler          = bundle["scaler"]
        FEATURES        = bundle["features"]
        MODEL_NAME      = bundle.get("model_name", "Random Forest")
        results_with    = bundle.get("results_with", {})
        results_without = bundle.get("results_without", {})
        cv_mean         = bundle.get("cv_mean", 0.0)
        cv_std          = bundle.get("cv_std",  0.0)
        print(f"[OK] Model loaded: {MODEL_NAME}")
    else:
        print("[WARN] Model not found. Run: python train.py")

load_bundle()

LABEL_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
LABEL_COLORS = {0: "danger", 1: "warning", 2: "success"}


def get_suggestions(pred_label, proba, row_dict):
    """Return academic advising notes based on model output."""
    suggestions = []
    if pred_label == 0:  # Dropout
        suggestions.append("Schedule an advising session with the academic department.")
        if row_dict.get("Tuition fees up to date", 1) == 0:
            suggestions.append("Check with the finance office regarding outstanding tuition dues.")
        if row_dict.get("Scholarship holder", 0) == 0:
            suggestions.append("Review eligibility criteria for institutional financial assistance.")
        if row_dict.get("Curricular units 2nd sem (approved)", 5) < 4:
            suggestions.append("Low credit completion observed; consider peer tutoring or reduced course load.")
        if row_dict.get("Debtor", 0) == 1:
            suggestions.append("Discuss an installment payment arrangement with student accounts.")
        suggestions.append("Utilize university learning support and counseling services.")
    elif pred_label == 1:  # Enrolled
        suggestions.append("Academic progress is steady, but consistent semester performance is needed.")
        if row_dict.get("Curricular units 2nd sem (grade)", 10) < 10:
            suggestions.append("Focus on core subject coursework to raise the cumulative GPA.")
        suggestions.append("Attend faculty office hours for guidance on difficult subject modules.")
        suggestions.append("Maintain attendance records to prevent missing internal assessments.")
    else:  # Graduate
        suggestions.append("Academic performance is strong and on track for timely degree completion.")
        suggestions.append("Explore final-year capstone topics, internship opportunities, or career placement.")
        suggestions.append("Maintain current study routines and semester grade point average.")
    return suggestions


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if model is None:
        return render_template('prediction.html',
                               error="Model not found. Run python train.py first.")

    if request.method == 'POST':
        try:
            # Collect form inputs
            row = {}
            field_map = {
                "Curricular units 1st sem (approved)": ("cu1_approved",   int,   5),
                "Curricular units 1st sem (grade)":    ("cu1_grade",      float, 12.0),
                "Curricular units 2nd sem (approved)": ("cu2_approved",   int,   5),
                "Curricular units 2nd sem (grade)":    ("cu2_grade",      float, 12.0),
                "Tuition fees up to date":             ("tuition_fees",   int,   1),
                "Scholarship holder":                  ("scholarship",    int,   0),
                "Debtor":                              ("debtor",         int,   0),
                "Age at enrollment":                   ("age",            int,   20),
                "Admission grade":                     ("admission_grade",float, 125.0),
                "Gender":                              ("gender",         int,   0),
            }

            for feature, (form_key, dtype, default) in field_map.items():
                if feature in FEATURES:
                    val = request.form.get(form_key, default)
                    converted_val = dtype(val)
                    # Convert Pakistan 4.0 GPA scale to European 0-20 scale internally for model compatibility
                    if form_key in ["cu1_grade", "cu2_grade"]:
                        converted_val = (converted_val / 4.0) * 20.0
                    # Convert Intermediate / Entry Test percentage (0-100%) to 0-200 scale internally
                    elif form_key == "admission_grade":
                        converted_val = (converted_val / 100.0) * 200.0
                    row[feature] = converted_val

            # Build input DataFrame with exact feature names to maintain pipeline compatibility
            input_df = pd.DataFrame([row], columns=FEATURES)
            input_scaled = scaler.transform(input_df)

            # Predict
            pred_class = model.predict(input_scaled)[0]
            proba      = model.predict_proba(input_scaled)[0]

            pred_label = LABEL_NAMES[pred_class]
            pred_color = LABEL_COLORS[pred_class]
            pred_pct   = round(proba[pred_class] * 100, 1)

            # Risk badge
            if proba[0] >= 0.6:
                risk_badge = ("High Risk", "danger")
            elif proba[0] >= 0.35:
                risk_badge = ("Medium Risk", "warning")
            else:
                risk_badge = ("Low Risk", "success")

            # Probabilities for chart
            prob_data = {
                "labels": ["Dropout", "Enrolled", "Graduate"],
                "prob_values": [float(round(p * 100, 1)) for p in proba],
                "colors": ["#dc3545", "#0d6efd", "#198754"],
            }

            suggestions = get_suggestions(pred_class, proba, row)

            return render_template('prediction.html',
                                   result=pred_label,
                                   pred_color=pred_color,
                                   pred_pct=pred_pct,
                                   risk_badge=risk_badge,
                                   prob_data=prob_data,
                                   suggestions=suggestions,
                                   submitted=True,
                                   row=row,
                                   features=FEATURES)

        except Exception as e:
            return render_template('prediction.html',
                                   error=f"Error: {str(e)}",
                                   features=FEATURES)

    return render_template('prediction.html', features=FEATURES)


@app.route('/results')
def results():
    # Prepare comparison table data
    rows = []
    for name in results_with:
        rows.append({
            "model":       name,
            "acc_smote":   round(results_with[name]["accuracy"] * 100, 1),
            "f1_smote":    round(results_with[name]["macro_f1"] * 100, 1),
            "acc_no":      round(results_without.get(name, {}).get("accuracy", 0) * 100, 1),
            "f1_no":       round(results_without.get(name, {}).get("macro_f1", 0) * 100, 1),
        })

    return render_template('results.html',
                           rows=rows,
                           best_model=MODEL_NAME,
                           cv_mean=round(cv_mean * 100, 1),
                           cv_std=round(cv_std * 100, 1))


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
