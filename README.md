🧠 NeuroAI — Personalized EEG-Based Cognitive & Emotional State Prediction for ASD

The rising global incidence of Autism Spectrum Disorder (ASD) has created a growing demand for innovative assistive technologies that can support the cognitive and emotional development of neurodiverse individuals. During our exploration of neurotechnology and machine learning, we identified a significant gap in real-time monitoring and interpretation of emotional and cognitive states in individuals with ASD. Existing systems often lack personalization, adaptability, and the ability to understand the unique neurological patterns associated with ASD.

NeuroAI was developed to address this challenge — a non-invasive, AI-powered platform designed to analyze and predict cognitive and emotional states using EEG (Electroencephalography) data. The system integrates two major modules:

🔹 Core Modules
🧠 Cognitive State Monitoring

Evaluates mental conditions such as:
Focus
Stress
Relaxation
Cognitive load
Deep learning models identify subtle EEG patterns associated with different cognitive states.

😊 Emotion Recognition

Classifies emotional states using CNN + LSTM architectures:
Happiness
Anxiety
Frustration
Calmness
Low attention

🎯 Therapy Response Prediction

One of NeuroAI’s unique features is identifying whether an individual is a responder or non-responder to therapeutic interventions.
This enables adaptive personalization where the system recommends:
Relaxation techniques
Mindfulness exercises
Focus-enhancing activities
Stress reduction tasks
These personalized suggestions help clinicians and caregivers choose effective therapy strategies.

📡 Real-Time Brain-State Feedback
NeuroAI provides real-time insights through an intuitive dashboard, making it a valuable tool for:
Clinicians
Therapists
Caregivers
Individuals with ASD
By delivering continuous, data-driven feedback, NeuroAI enhances therapy effectiveness and promotes emotional self-awareness.

📌 Keywords

EEG-based Cognitive Analysis, Autism Spectrum Disorder (ASD), Emotion Recognition, Machine Learning, Deep Learning, Cognitive State Monitoring, Personalized Treatment, NeuroAI, Predictive Modelling, Real-time Emotion Regulation

📂 Project Structure
NeuroAI/
│── data/                    # EEG dataset (not uploaded)
│── models/                  # Trained .h5 model files
│── src/
│   ├── preprocessing.py     # EEG preprocessing
│   ├── model.py             # CNN + LSTM model training
│   ├── inference.py         # Prediction script
│   ├── utils.py             # Helper functions
│── dashboard/               # Interface / visualization
│── requirements.txt         # Dependencies
│── README.md                # Documentation

🛠️ Installation
1️⃣ Clone the repo
git clone https://github.com/Siyachavan21/NeuroAI.git
cd NeuroAI

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Usage
🔹 Train the model
python src/model.py
🔹 Run prediction
python src/inference.py --file sample_eeg.csv
🔹 Launch dashboard
streamlit run dashboard/app.py

🧪 Models Used
CNN
LSTM / Bi-LSTM
EfficientNetB4 (optional)
ResNet50 (optional)
BatchNorm, Dropout, Dense layers

📈 Results
Accurate classification of emotional and cognitive states
Personalized therapy response prediction
Data-driven activity recommendations
Enhanced support for ASD therapy and regulation

👩‍💻 Author
Siya Chavan
GitHub: https://github.com/Siyachavan21
