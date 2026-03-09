# src/xai/explainer.py
# Explainable AI using SHAP — explains why the AI made each decision


import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExplanationResult:
    plain_english: str             # Human readable explanation
    feature_impacts: List[Tuple]   # [(feature_name, shap_value), ...]
    top_feature: str               # The single biggest driver
    top_impact: float              # Its SHAP value




# ── Training data ─────────────────────────────────────────────────────
# We train a simple model on synthetic accounting data.
# Features: [amount, keyword_score, vendor_known, is_recurring]
# This gives SHAP something to explain.


TRAINING_DATA = np.array([
    # amount  keyword  vendor_known  recurring
    [245.0,  0.9,     1,            1],   # Office Utilities
    [52.0,   0.8,     1,            0],   # Travel
    [1200.0, 0.7,     0,            0],   # Professional Services
    [18.50,  0.6,     1,            0],   # Meals
    [89.99,  0.85,    1,            1],   # Software
    [320.0,  0.5,     0,            0],   # Flagged (ambiguous)
    [78.0,   0.9,     1,            0],   # Office Supplies
    [445.0,  0.75,    1,            1],   # Office Utilities
])
TRAINING_LABELS = ['Office Utilities','Travel','Professional Services',
                   'Meals','Software','Review Required','Office Supplies','Office Utilities']


FEATURE_NAMES = ['Amount (£)', 'Keyword Match', 'Vendor Known', 'Recurring Expense']


# Train the model once
_le = LabelEncoder()
_y = _le.fit_transform(TRAINING_LABELS)
_model = RandomForestClassifier(n_estimators=20, random_state=42)
_model.fit(TRAINING_DATA, _y)


# Create SHAP explainer once
_explainer = shap.TreeExplainer(_model)




def explain_decision(
    amount: float,
    keyword_score: float,
    vendor_known: bool,
    is_recurring: bool = False
) -> ExplanationResult:
    '''
    Main function: explain why the model categorised this entry.


    Takes the same 4 features used in training and returns:
    - SHAP values per feature (how much each one influenced the decision)
    - Plain English summary
    '''
    # Prepare the input
    X = np.array([[
        amount,
        keyword_score,
        1 if vendor_known else 0,
        1 if is_recurring else 0
    ]])


    # Get SHAP values — one per feature, showing + or - influence
    shap_values = _explainer.shap_values(X)
    # shap_values[0] = values for class 0, shap_values[1] = class 1, etc.
    # We use the values for the predicted class
    predicted_class = _model.predict(X)[0]
    relevant_shap = shap_values[predicted_class][0]


    # Pair feature names with their SHAP values
    feature_impacts = list(zip(FEATURE_NAMES, relevant_shap.tolist()))
    # Sort by absolute impact — biggest driver first
    feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)


    top_feature, top_impact = feature_impacts[0]


    # ── Convert SHAP numbers to plain English ─────────────────────────
    explanation_parts = []
    for feat, val in feature_impacts:
        direction = 'strongly supported' if val > 0.1 else (
                    'slightly supported' if val > 0 else (
                    'slightly reduced confidence in' if val > -0.1 else
                    'reduced confidence in'))
        if abs(val) > 0.05:  # Only mention significant factors
            explanation_parts.append(f'{feat} {direction} this category (SHAP: {val:+.2f})')


    plain_english = 'AI Decision Explanation: ' + '; '.join(explanation_parts) + '.'


    return ExplanationResult(
        plain_english=plain_english,
        feature_impacts=feature_impacts,
        top_feature=top_feature,
        top_impact=top_impact
    )