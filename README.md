🧠 NeuroAI — EEG-Based Emotion & Cognitive State Prediction

NeuroAI is an advanced AI-powered assistive system designed to analyze and predict cognitive and emotional states of individuals with Autism Spectrum Disorder (ASD) using EEG (Electroencephalography) data.
The project uses Deep Learning (CNN + LSTM) to classify emotions such as happiness, anxiety, stress, and confusion from EEG brainwave signals.

🚀 Features

📡 EEG Signal Preprocessing
Noise removal, filtering, normalization, segmentation.

🧬 Deep Learning Models
CNN for spatial features + LSTM for temporal patterns.

🧠 Emotion Classification
Predicts states like: happiness, anxiety, stress, confusion, low attention.

📂 Project Structure
NeuroAI/
│── data/                    # EEG dataset (not uploaded)
│── models/                  # Trained .h5 model files
│── src/
│   ├── preprocessing.py     # EEG preprocessing
│   ├── model.py             # CNN + LSTM model training
│   ├── inference.py         # Prediction script
│   ├── utils.py             # Helper functions
│── dashboard/               # Streamlit / custom dashboard
│── requirements.txt         # Dependencies
│── README.md                # Documentation

🛠️ Installation
1️⃣ Clone the repo
git clone https://github.com/Siyachavan21/NeuroAI.git
cd NeuroAI

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ How to Use
🔹 Train the model
python src/model.py
🔹 Run prediction
python src/inference.py --file sample_eeg.csv
🔹 Launch dashboard
streamlit run dashboard/app.py

🧪 Models Used
EfficientNetB4
ResNet50
CNN Layers
LSTM / Bi-LSTM
BatchNorm, Dropout, Dense Layers

📈 Results
High accuracy in multi-class emotion prediction
Transfer learning improved performance
Stable predictions across subjects

👩‍💻 Author
Siya Chavan
GitHub: https://github.com/Siyachavan21
