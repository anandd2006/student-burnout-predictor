# 🎓 Student Burnout Predictor

An AI-powered web application that helps students track daily habits, predict burnout risk, and receive personalised wellbeing advice — built with Python, scikit-learn and Streamlit.

---

## 📋 Project Overview

Students often push themselves to the limit without realising the cumulative toll on their mental health. This tool acts as a **personal AI wellbeing assistant** that:

- Accepts daily habit data (sleep, study, stress, exercise, etc.)
- Predicts a **Burnout Score (0–100)** and a **Risk Level** (Low / Medium / High)
- Tracks scores over time and visualises trends
- Triggers **smart warnings** when concerning patterns emerge
- Generates a **Weekly Burnout Report** exportable as CSV or PDF
- Explains which habits contribute most to burnout via **feature importance**

---

## 🗂 Folder Structure

```
student_burnout_predictor/
├── data/
│   ├── burnout_dataset.csv     # Synthetic training data (auto-generated)
│   └── student_log.csv         # Your personal daily log (auto-created)
├── models/
│   ├── burnout_model.pkl       # Trained ML model
│   ├── scaler.pkl              # Feature scaler
│   ├── label_encoder.pkl       # Label encoder
│   └── metrics.json            # Model evaluation metrics
├── src/
│   ├── data_generator.py       # Synthetic dataset generator
│   ├── train_model.py          # Model training script
│   └── predict.py              # Prediction logic + advice engine
├── app/
│   └── app.py                  # Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate the dataset (optional — auto-runs during training)

```bash
python src/data_generator.py
```

### 3. Train the model

```bash
python src/train_model.py
```

This will:
- Load (or generate) the dataset
- Train Logistic Regression, Random Forest and Decision Tree models
- Select the best model by cross-validation accuracy
- Save the model, scaler, encoder and evaluation metrics to `models/`

### 4. Run a single prediction from the CLI

```bash
python src/predict.py
```

### 5. Launch the web app

```bash
streamlit run app/app.py
```

Open your browser at `http://localhost:8501`.

---

## 🤖 How the Model Works

| Step | Detail |
|---|---|
| **Data** | 1 000 synthetic student records with realistic noise |
| **Features** | Sleep, Study, Stress, Assignments, Break, Exercise |
| **Target** | Burnout Level: Low / Medium / High |
| **Models tried** | Logistic Regression, Random Forest ✅, Decision Tree |
| **Selection** | 5-fold cross-validation accuracy |
| **Burnout Score** | Probability-weighted score (0–100) |

### Burnout Score Formula

```
score = P(Low)×0 + P(Medium)×50 + P(High)×100
```

---

## 📱 App Pages

| Page | Description |
|---|---|
| 🏠 Home | Overview + active health warnings |
| 📋 Log Daily Data | Record habits + auto-save to CSV |
| 🔮 Predict Burnout | Instant score + personalised tips |
| 📈 Dashboard | Trend graphs, scatter plots, KPIs |
| 📊 Weekly Report | 7-day summary + PDF/CSV export |
| 🧪 Model Metrics | Accuracy, confusion matrix, feature importance |

---

## ⚡ Smart Warning System

The app monitors your log and fires alerts when:

- Burnout score > 70 for **3 consecutive days**
- Sleep < 5 h for **3 consecutive days**
- Stress > 8 for **3 consecutive days**
- Break time < 0.5 h for **3 consecutive days**

---

## 🔮 Future Improvements

- Real student survey data to replace synthetic training data
- Deep learning models (LSTM) for sequential burnout prediction
- Mobile app version (React Native or Flutter)
- Integration with Google Calendar for automatic workload detection
- AI chatbot for real-time burnout counselling
- Wearable device integration (sleep trackers, heart-rate monitors)
- University-wide anonymised dashboards for student support teams

---

## 🔒 Security & Privacy

In a production university deployment, the following safeguards would be required:

- **Data encryption** at rest and in transit (AES-256, TLS 1.3)
- **Secure authentication** (OAuth2 / SSO via university accounts)
- **Data anonymisation** — no PII should be stored alongside health metrics
- **Ethical AI audit** — regular bias checks on the model
- **GDPR / FERPA compliance** for student data protection
- **Right to deletion** — students can erase all their data at any time

> ⚠️ This application is a **demonstration tool**. Do not use it as a substitute for professional mental health advice. If you are struggling, please contact your university counselling service.

---

## 📸 Example Output

```
Burnout Score: 72 / 100
Risk Level:    HIGH 🚨

Personalised Suggestions:
  😴 Increase sleep to at least 7 hours to reduce burnout risk.
  😰 Consider meditation or reducing your workload.
  📚 Introduce Pomodoro breaks to your study sessions.
```

---

*Built with ❤️ using Python · scikit-learn · Streamlit*
