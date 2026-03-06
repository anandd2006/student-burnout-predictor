"""
data_generator.py
-----------------
Generates a realistic synthetic dataset of 1000 students for training the
Student Burnout Predictor model.

Run this file directly to regenerate the dataset:
    python src/data_generator.py
"""

import numpy as np
import pandas as pd
import os

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)


def generate_dataset(n_samples: int = 1000, save_path: str = "data/burnout_dataset.csv") -> pd.DataFrame:
    """
    Generate a synthetic student burnout dataset with realistic correlations
    and controlled random noise so the ML model must learn patterns rather
    than memorise strict rules.

    Parameters
    ----------
    n_samples : int
        Number of student daily records to generate (default 1 000).
    save_path : str
        Where to write the resulting CSV file.

    Returns
    -------
    pd.DataFrame
        The generated dataset.
    """

    # ── Raw feature sampling ─────────────────────────────────────────────────

    # Sleep hours  (3 – 10 h, skewed toward insufficient sleep)
    sleep_hours = np.clip(np.random.normal(loc=6.5, scale=1.5, size=n_samples), 3, 10).round(1)

    # Study hours  (1 – 14 h, skewed toward heavy studying)
    study_hours = np.clip(np.random.normal(loc=7.0, scale=2.5, size=n_samples), 1, 14).round(1)

    # Stress level (1 – 10, tends toward moderate-high)
    stress_level = np.clip(np.random.normal(loc=6.0, scale=2.0, size=n_samples), 1, 10).round(1)

    # Assignments due today (0 – 7)
    assignments_due = np.random.randint(0, 8, size=n_samples)

    # Break time (0 – 4 h)
    break_hours = np.clip(np.random.exponential(scale=1.2, size=n_samples), 0, 4).round(1)

    # Exercise (0 – 90 min)
    exercise_minutes = np.clip(np.random.exponential(scale=20, size=n_samples), 0, 90).round(0)

    # ── Burnout score (0 – 100) ──────────────────────────────────────────────
    # Each feature contributes additively; weights reflect real research findings.

    score = (
        (10 - sleep_hours)      * 4.5   # less sleep  → higher score
        + study_hours           * 2.5   # more study  → higher score
        + stress_level          * 5.0   # stress is the strongest driver
        + assignments_due       * 3.0   # deadline pressure
        - break_hours           * 3.5   # breaks are protective
        - exercise_minutes      * 0.2   # exercise is protective
        + np.random.normal(0, 8, n_samples)  # realistic noise
    )

    # Normalise to 0 – 100
    score = (score - score.min()) / (score.max() - score.min()) * 100
    burnout_score = np.clip(score, 0, 100).round(1)

    # ── Categorical label ────────────────────────────────────────────────────
    def score_to_label(s: float) -> str:
        if s <= 30:
            return "Low"
        elif s <= 60:
            return "Medium"
        else:
            return "High"

    burnout_level = [score_to_label(s) for s in burnout_score]

    # ── Assemble DataFrame ───────────────────────────────────────────────────
    df = pd.DataFrame({
        "sleep_hours":      sleep_hours,
        "study_hours":      study_hours,
        "stress_level":     stress_level,
        "assignments_due":  assignments_due,
        "break_hours":      break_hours,
        "exercise_minutes": exercise_minutes,
        "burnout_score":    burnout_score,
        "burnout_level":    burnout_level,
    })

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅  Dataset saved → {save_path}  ({len(df)} rows)")
    print(df["burnout_level"].value_counts())
    return df


if __name__ == "__main__":
    generate_dataset()
