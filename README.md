# Swiss Dialect AI - Streamlit App

A voice-based age prediction system for Swiss German speakers using acoustic embeddings and machine learning.

## 🎯 Features

- **Real-time voice recording** via browser microphone
- **Baseline model:** Predicts across 6 age categories (Teens, Twenties, Thirties, Forties, Sixties, Seventies)
- **Research model:** Binary classification (Young vs. Old)
- **Confidence visualization:** Shows prediction probabilities for both models
- **Professional UI:** Branded with Swiss national colors

## 📋 Requirements

See `requirements.txt` for all dependencies. Main packages:

- `streamlit` - Web app framework
- `torch` - Deep learning
- `transformers` - Wav2Vec2 for audio embeddings
- `lightgbm` - Gradient boosting classifiers
- `librosa` - Audio processing

## 🚀 Installation & Local Testing

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure model files are present:**
   - `baseline_results.pkl` (6-class classifier)
   - `research_results.pkl` (binary classifier)

3. **Run locally:**

   ```bash
   streamlit run app.py
   ```

4. **Access:** Open http://localhost:8501 in your browser

## ☁️ Deploying on Streamlit Community Cloud

1. **Push to GitHub** with this folder structure:

   ```
   your-repo/
   ├── age_prediction/
   │   └── dialect_app/
   │       ├── app.py
   │       ├── requirements.txt
   │       ├── baseline_results.pkl
   │       ├── research_results.pkl
   │       └── .streamlit/
   │           └── config.toml
   ```

2. **Create Streamlit account** at [share.streamlit.io](https://share.streamlit.io)

3. **Deploy:**
   - Click "New app"
   - Select your repository
   - Choose main branch
   - Set main file path to `age_prediction/dialect_app/app.py`
   - Click "Deploy"

## 🎤 Usage

1. Click **"Start Recording"** button
2. Speak naturally in Swiss German for 5-10 seconds
3. Click **"Stop Recording"**
4. Click **"🔍 Analyze Acoustic Features"**
5. View predictions from both models with confidence scores

## 🔍 Model Details

### Baseline Model

- **Type:** LightGBM classifier with 500 estimators
- **Input:** Wav2Vec2 embeddings (1024-dimensional)
- **Output:** 6 age categories
- **Classes:** Teens (0), Twenties (1), Thirties (2), Forties (3), Sixties (4), Seventies (5)

### Research Model

- **Type:** LightGBM classifier with 1000 estimators (binary)
- **Input:** Wav2Vec2 embeddings
- **Output:** Young vs. Old
- **Classes:** Young (0) = Teens/Twenties, Old (1) = Sixties/Seventies

## 🧠 Audio Processing

- **Sampling rate:** 16 kHz
- **Feature extractor:** Facebook's Wav2Vec2-Large-XLSR-53-German
- **Feature aggregation:** Mean pooling over time dimension
- **Max duration:** 10 seconds

## 📊 Dataset

**STT4SG-350 Corpus**

- Swiss German speech recordings
- Stratified by age groups and regions
- Used for training and evaluation

## 🎓 Project

**Introduction to NLP (Spring 2026)**  
Acoustic Fingerprints of Generational Change in Swiss German

## 📝 Notes

- Best results with clear audio in a quiet environment
- Model was trained on Swiss German dialect
- Works best with native speakers
- Minimum audio length: 1-2 seconds recommended

---

**Colors:** #156082 (Primary), #E67E22 (Accent)
