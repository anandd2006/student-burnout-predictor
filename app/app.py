"""
app.py  —  Student Burnout Predictor  (Streamlit Web App)
----------------------------------------------------------
A multi-page application that lets students:
  • Log daily habits
  • Predict burnout risk (with score)
  • Track trends over time
  • View a visual dashboard
  • Generate and export a weekly report

Run:
    streamlit run app/app.py
"""

import os
import sys
import json
import datetime
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
import subprocess
import os

# Auto-train model if not already trained
if not os.path.exists("models/burnout_model.pkl"):
    subprocess.run(["python", "src/train_model.py"])

# ── Make sure src/ is on the Python path ─────────────────────────────────────
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC_DIR))

from predict import predict_burnout, personalized_advice   # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "models")
DATA_DIR     = os.path.join(BASE_DIR, "data")
LOG_PATH     = os.path.join(DATA_DIR, "student_log.csv")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

LOG_COLS = [
    "date", "sleep_hours", "study_hours", "stress_level",
    "assignments_due", "break_hours", "exercise_minutes", "burnout_score", "burnout_level",
]


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def risk_color(label: str) -> str:
    return {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}.get(label, "#999")


def risk_emoji(label: str) -> str:
    return {"Low": "✅", "Medium": "⚠️", "High": "🚨"}.get(label, "")


def load_log() -> pd.DataFrame:
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH, parse_dates=["date"])
        return df
    return pd.DataFrame(columns=LOG_COLS)


def append_log(row: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_log()
    new_row = pd.DataFrame([row])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(LOG_PATH, index=False)


def load_metrics() -> dict | None:
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def check_smart_warnings(df: pd.DataFrame) -> list[str]:
    """Return warning messages triggered by recent trend patterns."""
    warnings = []
    if len(df) < 3:
        return warnings
    recent = df.tail(3)

    if (recent["burnout_score"] > 70).all():
        warnings.append("🚨  Your burnout score has been above 70 for 3 consecutive days. Please consider taking a recovery day!")
    if (recent["sleep_hours"] < 5).all():
        warnings.append("😴  Severe sleep deprivation: fewer than 5 h of sleep for 3 days in a row. Your brain needs rest!")
    if (recent["stress_level"] > 8).all():
        warnings.append("😰  Sustained high stress (>8) for 3 days. Please speak with a counsellor or trusted person.")
    if (recent["break_hours"] < 0.5).all():
        warnings.append("☕  You've had almost no breaks for 3 days. Short breaks are essential for mental clarity.")
    return warnings


# ════════════════════════════════════════════════════════════════════════════
# Streamlit page config
# ════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header   { font-size:2.4rem; font-weight:700; color:#1a1a2e; }
    .sub-header    { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
    .score-box     { border-radius:12px; padding:24px; text-align:center; margin:10px 0; }
    .metric-card   { background:#f8f9fa; border-radius:10px; padding:16px; margin:6px 0; }
    .advice-card   { background:#eaf4fb; border-left:4px solid #3498db;
                     border-radius:6px; padding:12px 16px; margin:6px 0; }
    .warning-card  { background:#fff3cd; border-left:4px solid #f39c12;
                     border-radius:6px; padding:12px 16px; margin:6px 0; }
    .success-card  { background:#d4edda; border-left:4px solid #2ecc71;
                     border-radius:6px; padding:12px 16px; margin:6px 0; }
    div[data-testid="stSidebarNav"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/color/96/student-center.png", width=80)
    st.markdown("## 🎓 Burnout Predictor")
    st.markdown("*Your AI wellbeing assistant*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠  Home", "📋  Log Daily Data", "🔮  Predict Burnout",
         "📈  Dashboard", "📊  Weekly Report", "🧪  Model Metrics"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("v2.0  |  Built with ❤️ & scikit-learn")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    st.markdown('<p class="main-header">🎓 Student Burnout Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">An AI-powered wellbeing assistant for students — track habits, predict burnout, and take control of your mental health.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔮 Predict Burnout**\n\nEnter your daily habits and get an instant burnout score with personalised advice.")
    with col2:
        st.warning("**📋 Log Daily Data**\n\nRecord your habits every day and build a personal wellbeing timeline.")
    with col3:
        st.success("**📈 View Dashboard**\n\nVisualise trends over time: sleep, stress, burnout score and more.")

    st.divider()

    # Smart warnings from log
    df_log = load_log()
    warnings = check_smart_warnings(df_log)
    if warnings:
        st.subheader("⚠️ Active Health Warnings")
        for w in warnings:
            st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    st.subheader("How It Works")
    st.markdown("""
1. **Log** your daily sleep, study, stress and break data.
2. **Predict** your burnout risk using our ML model (Random Forest).
3. **Track** your burnout score over days and weeks.
4. **Act** on personalised suggestions to protect your mental health.
    """)

    st.subheader("Burnout Score Scale")
    cols = st.columns(3)
    for col, (label, rng, color) in zip(cols, [
        ("Low Risk",    "0 – 30",   "#2ecc71"),
        ("Medium Risk", "31 – 60",  "#f39c12"),
        ("High Risk",   "61 – 100", "#e74c3c"),
    ]):
        col.markdown(
            f'<div class="score-box" style="background:{color}20;border:2px solid {color};">'
            f'<h3 style="color:{color};">{label}</h3><p style="font-size:1.4rem;">{rng}</p></div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Log Daily Data
# ════════════════════════════════════════════════════════════════════════════

elif page == "📋  Log Daily Data":
    st.header("📋 Log Today's Data")
    st.markdown("Record your daily habits to track your wellbeing over time.")

    with st.form("log_form"):
        col1, col2 = st.columns(2)
        with col1:
            log_date   = st.date_input("Date", value=datetime.date.today())
            sleep_h    = st.slider("😴 Sleep Hours",          3.0, 10.0, 7.0, 0.5)
            study_h    = st.slider("📚 Study Hours",          1.0, 14.0, 6.0, 0.5)
            stress_lvl = st.slider("😰 Stress Level (1-10)",  1,   10,   5)
        with col2:
            assign     = st.number_input("📝 Assignments Due", 0, 10, 2)
            break_h    = st.slider("☕ Break Hours",           0.0, 4.0, 1.0, 0.25)
            exercise   = st.slider("🏃 Exercise (minutes)",   0,   90,  20)

        submitted = st.form_submit_button("💾 Save Entry", use_container_width=True)

    if submitted:
        result = predict_burnout(sleep_h, study_h, stress_lvl, assign, break_h, exercise)
        row = {
            "date":              str(log_date),
            "sleep_hours":       sleep_h,
            "study_hours":       study_h,
            "stress_level":      stress_lvl,
            "assignments_due":   assign,
            "break_hours":       break_h,
            "exercise_minutes":  exercise,
            "burnout_score":     result["score"],
            "burnout_level":     result["label"],
        }
        append_log(row)

        color = risk_color(result["label"])
        st.markdown(
            f'<div class="score-box" style="background:{color}20;border:2px solid {color};">'
            f'<h2 style="color:{color};">Entry saved! Burnout Score: {result["score"]}/100 — {result["label"].upper()} {risk_emoji(result["label"])}</h2>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Show existing log
    df_log = load_log()
    if not df_log.empty:
        st.subheader("📅 Your Log History")
        st.dataframe(df_log.sort_values("date", ascending=False).head(30), use_container_width=True)

        # Export CSV
        csv_bytes = df_log.to_csv(index=False).encode()
        st.download_button("⬇️ Export Log as CSV", csv_bytes, "burnout_log.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Predict Burnout
# ════════════════════════════════════════════════════════════════════════════

elif page == "🔮  Predict Burnout":
    st.header("🔮 Predict Your Burnout Risk")
    st.markdown("Enter today's habits and get an instant AI prediction.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Enter Your Habits")
        sleep_h    = st.slider("😴 Sleep Hours",          3.0, 10.0, 7.0, 0.5)
        study_h    = st.slider("📚 Study Hours",          1.0, 14.0, 6.0, 0.5)
        stress_lvl = st.slider("😰 Stress Level (1-10)",  1,   10,   5)
        assign     = st.number_input("📝 Assignments Due", 0, 10, 2)
        break_h    = st.slider("☕ Break Hours",           0.0, 4.0, 1.0, 0.25)
        exercise   = st.slider("🏃 Exercise (minutes)",   0,   90,  20)
        predict_btn = st.button("🔮 Predict Burnout", use_container_width=True, type="primary")

    with col2:
        if predict_btn:
            result = predict_burnout(sleep_h, study_h, stress_lvl, assign, break_h, exercise)
            label  = result["label"]
            score  = result["score"]
            color  = risk_color(label)
            emoji  = risk_emoji(label)

            # Score display
            st.markdown(
                f'<div class="score-box" style="background:{color}20;border:3px solid {color};">'
                f'<h1 style="color:{color};font-size:3rem;">{score}<span style="font-size:1.5rem;">/100</span></h1>'
                f'<h2 style="color:{color};">Burnout Risk: {label.upper()} {emoji}</h2>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability gauge
            st.subheader("Class Probabilities")
            for cls, p in result["probabilities"].items():
                c = risk_color(cls)
                st.markdown(f"**{cls}**: {p*100:.1f}%")
                st.progress(p)

            # Risk message
            st.subheader("What This Means")
            if label == "High":
                st.error("⚠️ You may be experiencing serious burnout. Please prioritise your wellbeing immediately.")
            elif label == "Medium":
                st.warning("⚠️ You're at moderate risk. Monitor your schedule and make small improvements.")
            else:
                st.success("✅ You're maintaining a healthy balance. Keep it up!")

            # Personalised advice
            tips = personalized_advice(sleep_h, study_h, stress_lvl, break_h, exercise, label)
            if tips:
                st.subheader("💡 Personalised Suggestions")
                for tip in tips:
                    st.markdown(f'<div class="advice-card">{tip}</div>', unsafe_allow_html=True)

            # Mini chart: feature contribution
            st.subheader("Your Habit Overview")
            fig, ax = plt.subplots(figsize=(5, 3))
            features_display = {
                "Sleep (h)":    sleep_h,
                "Study (h)":    study_h,
                "Stress /10":   stress_lvl,
                "Assignments":  assign,
                "Break (h)":    break_h,
                "Exercise(min)":exercise,
            }
            bars = ax.barh(list(features_display.keys()), list(features_display.values()),
                           color=color, alpha=0.75, edgecolor="white")
            ax.set_xlabel("Value")
            ax.set_title("Today's Habits at a Glance")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        else:
            st.info("👈 Set your habits on the left and click **Predict Burnout**.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Dashboard
# ════════════════════════════════════════════════════════════════════════════

elif page == "📈  Dashboard":
    st.header("📈 Personal Burnout Dashboard")

    df_log = load_log()
    if df_log.empty:
        st.info("No data yet. Log some daily entries first!")
        st.stop()

    df_log = df_log.sort_values("date")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Burnout Score", f"{df_log['burnout_score'].mean():.1f}/100")
    k2.metric("Avg Sleep",         f"{df_log['sleep_hours'].mean():.1f} h")
    k3.metric("Avg Stress",        f"{df_log['stress_level'].mean():.1f}/10")
    k4.metric("Entries Logged",    len(df_log))

    # ── Active smart warnings ─────────────────────────────────────────────
    warnings = check_smart_warnings(df_log)
    if warnings:
        st.subheader("⚠️ Smart Warnings")
        for w in warnings:
            st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Burnout score over time ───────────────────────────────────────────
    st.subheader("📉 Burnout Score Over Time")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.fill_between(df_log["date"], df_log["burnout_score"], alpha=0.2, color="#e74c3c")
    ax.plot(df_log["date"], df_log["burnout_score"], color="#e74c3c", linewidth=2, marker="o", markersize=5)
    ax.axhline(30,  color="#2ecc71", linestyle="--", alpha=0.6, label="Low threshold")
    ax.axhline(60,  color="#f39c12", linestyle="--", alpha=0.6, label="Medium threshold")
    ax.set_ylabel("Burnout Score")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    col1, col2 = st.columns(2)

    # ── Sleep vs Burnout score ────────────────────────────────────────────
    with col1:
        st.subheader("😴 Sleep vs Burnout Score")
        fig, ax = plt.subplots(figsize=(5, 4))
        colors = [risk_color(l) for l in df_log["burnout_level"]]
        ax.scatter(df_log["sleep_hours"], df_log["burnout_score"], c=colors, alpha=0.75, edgecolors="white", s=70)
        ax.set_xlabel("Sleep Hours")
        ax.set_ylabel("Burnout Score")
        patches = [mpatches.Patch(color=risk_color(l), label=l) for l in ["Low","Medium","High"]]
        ax.legend(handles=patches, fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Stress distribution ───────────────────────────────────────────────
    with col2:
        st.subheader("😰 Stress Level Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df_log["stress_level"], bins=10, color="#9b59b6", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Stress Level (1-10)")
        ax.set_ylabel("Frequency")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # ── Study hours histogram ─────────────────────────────────────────────
    st.subheader("📚 Study Hours & Sleep Trends")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(df_log["date"], df_log["study_hours"], color="#3498db", marker="o", markersize=4)
    axes[0].set_ylabel("Study Hours")
    axes[0].set_title("Study Hours Over Time")
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(df_log["date"], df_log["sleep_hours"], color="#1abc9c", marker="o", markersize=4)
    axes[1].set_ylabel("Sleep Hours")
    axes[1].set_title("Sleep Hours Over Time")
    axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Weekly Report
# ════════════════════════════════════════════════════════════════════════════

elif page == "📊  Weekly Report":
    st.header("📊 Weekly Burnout Report")

    df_log = load_log()
    if df_log.empty:
        st.info("No data yet. Log at least 7 days of data to generate a weekly report.")
        st.stop()

    df_log = df_log.sort_values("date")
    # Use last 7 entries as the "week"
    week_df = df_log.tail(7).copy()

    st.subheader("📅 Week at a Glance")
    st.dataframe(week_df.set_index("date"), use_container_width=True)

    st.divider()

    # Stats
    avg_score  = week_df["burnout_score"].mean()
    max_stress_day = week_df.loc[week_df["stress_level"].idxmax(), "date"]
    min_sleep_day  = week_df.loc[week_df["sleep_hours"].idxmin(),  "date"]
    avg_study  = week_df["study_hours"].mean()
    high_days  = (week_df["burnout_level"] == "High").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Burnout Score", f"{avg_score:.1f}/100")
    c2.metric("High-Risk Days",         f"{high_days} / {len(week_df)}")
    c3.metric("Avg Study Hours",        f"{avg_study:.1f} h")

    st.markdown(f"""
| Metric                | Value |
|----------------------|-------|
| Highest-Stress Day   | {max_stress_day} |
| Lowest-Sleep Day     | {min_sleep_day} |
| Avg Sleep            | {week_df['sleep_hours'].mean():.1f} h |
| Avg Exercise         | {week_df['exercise_minutes'].mean():.0f} min |
| Avg Break Hours      | {week_df['break_hours'].mean():.1f} h |
    """)

    # Recommendations
    st.subheader("📋 AI Recommendations for This Week")
    rec = []
    if avg_score > 60:
        rec.append("🚨 Your average burnout score is high. Plan at least one full rest day next week.")
    if week_df["sleep_hours"].mean() < 6.5:
        rec.append("😴 Improve your sleep schedule — aim for 7-8 h per night.")
    if week_df["exercise_minutes"].mean() < 20:
        rec.append("🏃 Increase physical activity — even 20-30 min walks make a measurable difference.")
    if week_df["break_hours"].mean() < 1:
        rec.append("☕ Schedule structured breaks — try the Pomodoro method (25 min work / 5 min break).")
    if not rec:
        rec.append("✅ Great week! Keep up your healthy habits.")
    for r in rec:
        st.markdown(f'<div class="advice-card">{r}</div>', unsafe_allow_html=True)

    # Burnout chart for the week
    st.subheader("Burnout Score This Week")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bar_colors = [risk_color(l) for l in week_df["burnout_level"]]
    ax.bar(week_df["date"].astype(str), week_df["burnout_score"], color=bar_colors, edgecolor="white")
    ax.set_ylabel("Burnout Score")
    ax.set_ylim(0, 100)
    ax.axhline(30, linestyle="--", color="#2ecc71", alpha=0.6)
    ax.axhline(60, linestyle="--", color="#f39c12", alpha=0.6)
    plt.xticks(rotation=25, ha="right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Export CSV
    csv_bytes = week_df.to_csv(index=False).encode()
    st.download_button("⬇️ Export Weekly Report (CSV)", csv_bytes, "weekly_report.csv", "text/csv")

    # PDF export using reportlab
    if st.button("📄 Generate PDF Report"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib           import colors as rl_colors
            from reportlab.platypus      import (SimpleDocTemplate, Paragraph, Spacer,
                                                  Table, TableStyle)
            from reportlab.lib.styles    import getSampleStyleSheet
            import io

            buf = io.BytesIO()
            doc = SimpleDocTemplate(buf, pagesize=A4)
            styles = getSampleStyleSheet()
            story  = []

            story.append(Paragraph("🎓 Student Burnout Weekly Report", styles["Title"]))
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"Generated: {datetime.date.today()}", styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph(f"Average Burnout Score: {avg_score:.1f}/100", styles["Heading2"]))
            story.append(Paragraph(f"High-Risk Days: {high_days}/{len(week_df)}", styles["Normal"]))
            story.append(Paragraph(f"Average Study Hours: {avg_study:.1f} h", styles["Normal"]))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Recommendations", styles["Heading2"]))
            for r in rec:
                story.append(Paragraph(f"• {r}", styles["Normal"]))
            story.append(Spacer(1, 12))

            # Table of week data
            table_data = [["Date","Sleep","Study","Stress","Score","Level"]]
            for _, row in week_df.iterrows():
                table_data.append([
                    str(row["date"])[:10], row["sleep_hours"], row["study_hours"],
                    row["stress_level"], row["burnout_score"], row["burnout_level"],
                ])
            t = Table(table_data)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), rl_colors.HexColor("#3498db")),
                ("TEXTCOLOR",  (0,0), (-1,0), rl_colors.white),
                ("GRID",       (0,0), (-1,-1), 0.5, rl_colors.grey),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [rl_colors.white, rl_colors.HexColor("#f8f9fa")]),
            ]))
            story.append(t)

            doc.build(story)
            buf.seek(0)
            st.download_button("⬇️ Download PDF", buf, "weekly_burnout_report.pdf", "application/pdf")
        except ImportError:
            st.error("Please install reportlab:  pip install reportlab")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: Model Metrics
# ════════════════════════════════════════════════════════════════════════════

elif page == "🧪  Model Metrics":
    st.header("🧪 Model Evaluation Dashboard")

    metrics = load_metrics()
    if not metrics:
        st.warning("No metrics found. Please run `python src/train_model.py` first.")
        st.stop()

    # Overview
    st.subheader("Best Model Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Best Model",  metrics["best_model"])
    c2.metric("Test Accuracy", f"{metrics['accuracy']*100:.2f}%")
    c3.metric("CV Accuracy",   f"{metrics['cv_score']*100:.2f}%")

    st.divider()

    # Model comparison bar chart
    st.subheader("Model Comparison")
    comp = metrics["model_comparison"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names   = list(comp.keys())
    acc_vals = [comp[n]["accuracy"] for n in names]
    cv_vals  = [comp[n]["cv_score"]  for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax.bar(x - w/2, acc_vals, w, label="Test Accuracy", color="#3498db", alpha=0.85)
    ax.bar(x + w/2, cv_vals,  w, label="CV Accuracy",   color="#2ecc71", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm     = np.array(metrics["confusion_matrix"])
    labels = metrics["classes"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Classification report
    st.subheader("Classification Report")
    report = metrics["classification_report"]
    rows = []
    for cls in labels:
        r = report.get(cls, {})
        rows.append({
            "Class":     cls,
            "Precision": round(r.get("precision", 0), 3),
            "Recall":    round(r.get("recall", 0),    3),
            "F1-Score":  round(r.get("f1-score", 0),  3),
            "Support":   int(r.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Class"), use_container_width=True)

    # Feature importance
    st.subheader("🔍 Feature Importance (Random Forest)")
    fi = metrics.get("feature_importance", {})
    if fi:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sorted_fi = dict(sorted(fi.items(), key=lambda x: x[1]))
        ax.barh(list(sorted_fi.keys()), list(sorted_fi.values()),
                color="#9b59b6", alpha=0.85, edgecolor="white")
        ax.set_xlabel("Importance")
        ax.set_title("Top Burnout Drivers")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.subheader("Ranked Burnout Causes")
        for rank, (feat, imp) in enumerate(sorted(fi.items(), key=lambda x: x[1], reverse=True), 1):
            st.markdown(f"**{rank}. {feat.replace('_',' ').title()}** — importance: `{imp:.4f}`")
