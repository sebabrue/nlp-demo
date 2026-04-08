import streamlit as st
import torch
import numpy as np
import io
import subprocess
import joblib
from transformers import AutoProcessor, Wav2Vec2Model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Swiss Dialect AI", page_icon="🇨🇭", layout="centered")

# --- MAPPING (Passe diese Liste an deine Kaggle-Klassen an!) ---
# Die Reihenfolge muss exakt der deines Trainings entsprechen (LabelEncoder.classes_)
AGE_CLASSES = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"] 

@st.cache_resource
def load_models():
    model_id = "facebook/wav2vec2-large-xlsr-53-german"
    processor = AutoProcessor.from_pretrained(model_id)
    wav_model = Wav2Vec2Model.from_pretrained(model_id).eval()
    
    # Lade dein Modell (Multiclass)
    data = joblib.load('research_results.pkl') 
    classifier = data['model'] if isinstance(data, dict) and 'model' in data else data
    return processor, wav_model, classifier

def load_audio_from_bytes_ffmpeg(audio_bytes, sr=16000):
    """
    Diese Funktion imitiert exakt deine Kaggle 'load_audio_ffmpeg' Funktion,
    nutzt aber Bytes anstatt eines Dateipfads.
    """
    cmd = [
        'ffmpeg', '-threads', '1', '-v', 'error', '-i', 'pipe:0', # pipe:0 liest von stdin
        '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1', '-'
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate(input=audio_bytes)
        if out:
            # Kaggle Logik: Normierung auf -1.0 bis 1.0
            return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return None
    except:
        return None

# --- UI START ---
processor, wav_model, classifier = load_models()

st.title("🇨🇭 Swiss German Age Predictor")
st.write("Live analysis using the exact Kaggle preprocessing pipeline.")

from streamlit_mic_recorder import mic_recorder
audio_data = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", key='recorder')

if audio_data:
    audio_bytes = audio_data['bytes']
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("🚀 Run Precise Analysis"):
        with st.spinner('Extracting Features (FFmpeg)...'):
            try:
                # 1. Laden mit FFmpeg (identisch zu Kaggle)
                aud = load_audio_from_bytes_ffmpeg(audio_bytes)
                
                if aud is not None and len(aud) > 1600:
                    # 2. Truncate auf 5 Sekunden (80000 Samples bei 16kHz)
                    aud = aud[:80000]
                    
                    # 3. Wav2Vec2 Feature Extraction
                    # Padding wird hier von dem Processor übernommen (wie in gpu_worker)
                    inputs = processor(aud, sampling_rate=16000, return_tensors="pt", padding=True)
                    
                    with torch.no_grad():
                        # 4. Mean Pooling (identisch zu Kaggle: .mean(dim=1))
                        outputs = wav_model(**inputs)
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    # 5. Prediction
                    prediction_idx = classifier.predict(emb)[0]
                    
                    # Mapping
                    if isinstance(prediction_idx, (int, np.integer)):
                        result_label = AGE_CLASSES[prediction_idx]
                    else:
                        result_label = prediction_idx

                    st.divider()
                    st.balloons()
                    st.success(f"### Predicted Age Group: **{result_label}**")
                else:
                    st.warning("Audio too short or not recognized. Please record again.")
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")

st.divider()
st.caption("Acoustic Fingerprinting @ Spring 2026. Models: Wav2Vec2-XLSR-53 + LightGBM.")