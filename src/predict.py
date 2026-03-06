"""
predict.py
----------
Loads the trained model and makes a burnout prediction for a single student.

Usage:
    python src/predict.py
"""

import joblib
import numpy as np
import os

MODEL_PATH   = "models/burnout_model.pkl"
SCALER_PATH  = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

FEATURE_COLS = [
    "sleep_hours", "study_hours", "stress_level",
    "assignments_due", "break_hours", "exercise_minutes",
]


def load_artefacts():
    """Load model, scaler and label encoder from disk."""
    model   = joblib.load(MODEL_PATH)
    scaler  = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder


def predict_burnout(
    sleep_hours: float,
    study_hours: float,
    stress_level: float,
    assignments_due: int,
    break_hours: float,
    exercise_minutes: float,
) -> dict:
    """
    Predict burnout risk and compute a 0-100 burnout score.

    Returns
    -------
    dict with keys: label, score, probabilities
    """
    model, scaler, encoder = load_artefacts()

    # Build feature vector
    features = np.array([[
        sleep_hours, study_hours, stress_level,
        assignments_due, break_hours, exercise_minutes,
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict class probabilities
    proba = model.predict_proba(features_scaled)[0]   # [P(High), P(Low), P(Medium)] — order depends on encoder
    label_idx = np.argmax(proba)
    label     = encoder.inverse_transform([label_idx])[0]

    # Map probabilities to class names
    class_proba = {cls: round(float(p), 4) for cls, p in zip(encoder.classes_, proba)}

    # Burnout score: weighted sum (High→100, Medium→50, Low→0)
    weight_map = {"Low": 0, "Medium": 50, "High": 100}
    score = sum(class_proba[cls] * weight_map[cls] for cls in encoder.classes_)
    score = round(score, 1)

    return {
        "label":         label,
        "score":         score,
        "probabilities": class_proba,
    }


def personalized_advice(
    sleep_hours: float,
    study_hours: float,
    stress_level: float,
    break_hours: float,
    exercise_minutes: float,
    label: str,
) -> list[str]:
    """Generate up to 5 personalised suggestions based on the student's inputs."""
    advice = []

    if sleep_hours < 6:
        advice.append(f"😴  Your sleep ({sleep_hours}h) is below the recommended 7-9 h. Try going to bed 30 min earlier each night.")
    if study_hours > 9:
        advice.append(f"📚  Studying {study_hours}h/day is intense. Try the Pomodoro technique: 25 min study → 5 min break.")
    if stress_level > 7:
        advice.append(f"😰  Stress level {stress_level}/10 is high. Consider mindfulness, deep breathing, or talking to a counsellor.")
    if break_hours < 1:
        advice.append(f"☕  You're taking only {break_hours}h of breaks. Short, regular breaks improve focus and reduce burnout.")
    if exercise_minutes < 20:
        advice.append(f"🏃  Only {exercise_minutes} min of exercise detected. Even a 20-min walk reduces cortisol significantly.")

    # Fallback for Low burnout
    if not advice and label == "Low":
        advice.append("✅  Great habits! Keep maintaining your current routine.")

    return advice[:5]   # cap at 5


# ── Demo run ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example student data
    inputs = dict(
        sleep_hours      = 5,
        study_hours      = 9,
        stress_level     = 8,
        assignments_due  = 4,
        break_hours      = 0.5,
        exercise_minutes = 10,
    )

    result = predict_burnout(**inputs)
    advice = personalized_advice(
        sleep_hours      = inputs["sleep_hours"],
        study_hours      = inputs["study_hours"],
        stress_level     = inputs["stress_level"],
        break_hours      = inputs["break_hours"],
        exercise_minutes = inputs["exercise_minutes"],
        label            = result["label"],
    )

    print("\n" + "=" * 40)
    print("  Burnout Prediction Result")
    print("=" * 40)
    print(f"  Burnout Score : {result['score']} / 100")
    print(f"  Risk Level    : {result['label'].upper()}")
    print(f"\n  Class Probabilities:")
    for cls, p in result["probabilities"].items():
        print(f"    {cls:<8} {p*100:.1f}%")
    print("\n  Personalized Advice:")
    for tip in advice:
        print(f"    • {tip}")
    print("=" * 40)
