import streamlit as st
import torch
import numpy as np
import io
import subprocess
import joblib
from transformers import AutoProcessor, Wav2Vec2Model
from streamlit_mic_recorder import mic_recorder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Swiss Dialect AI", page_icon="🇨🇭", layout="centered")

# Custom CSS for ZHAW-style branding
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #3498db; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION & MAPPING ---
# Ensure these match your LabelEncoder.classes_ from Kaggle!
AGE_CLASSES = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"] 

@st.cache_resource
def load_models():
    """Loads Wav2Vec2 and the trained Classifier with dictionary handling."""
    model_id = "facebook/wav2vec2-large-xlsr-53-german"
    processor = AutoProcessor.from_pretrained(model_id)
    wav_model = Wav2Vec2Model.from_pretrained(model_id).eval()
    
    # Load your .pkl file
    data = joblib.load('research_results.pkl') 
    
    # Dictionary fix: extract the actual model object
    if isinstance(data, dict):
        if 'model' in data:
            classifier = data['model']
        else:
            # Fallback: take the first value if 'model' key is missing
            classifier = list(data.values())[0] 
    else:
        classifier = data
        
    return processor, wav_model, classifier

def load_audio_from_bytes_ffmpeg(audio_bytes, sr=16000):
    """Replicates Kaggle FFmpeg loading: normalization to [-1, 1]."""
    cmd = [
        'ffmpeg', '-threads', '1', '-v', 'error', '-i', 'pipe:0', 
        '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1', '-'
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate(input=audio_bytes)
        if out:
            return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return None
    except Exception:
        return None

# --- UI LOGIC ---
st.title("🇨🇭 Swiss German Age Predictor")
st.markdown("### Acoustic Fingerprints of Generational Change")
st.write("Record your voice to let the AI predict your age group based on dialectal features.")

try:
    processor, wav_model, classifier = load_models()
except Exception as e:
    st.error(f"Initialization Error: {e}")

st.divider()

# Microphone Input
st.subheader("Step 1: Record your voice")
audio_data = mic_recorder(
    start_prompt="🔴 Start Recording", 
    stop_prompt="⏹️ Stop Recording", 
    key='recorder'
)

if audio_data:
    audio_bytes = audio_data['bytes']
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("🚀 Run Precise Analysis"):
        with st.spinner('Analyzing... (This may take a few seconds)'):
            try:
                # 1. Load via FFmpeg (Kaggle-style)
                aud = load_audio_from_bytes_ffmpeg(audio_bytes)
                
                if aud is not None and len(aud) > 1600:
                    # 2. Preprocessing: Cap at 5s (80,000 samples)
                    aud = aud[:80000]
                    
                    # 3. Wav2Vec2 Feature Extraction
                    inputs = processor(aud, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        outputs = wav_model(**inputs)
                        # Mean Pooling over time dimension
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                    
                    # 4. Prediction using the extracted classifier
                    prediction_idx = classifier.predict(emb)[0]
                    
                    # 5. Label Mapping
                    if isinstance(prediction_idx, (int, np.integer)):
                        result_label = AGE_CLASSES[prediction_idx]
                    else:
                        result_label = prediction_idx

                    # Result Display
                    st.divider()
                    st.balloons()
                    st.success(f"## Prediction: **{result_label} Years**")
                    st.info("The prediction is based on the phonetic leveling observed in your dialect.")
                else:
                    st.warning("Recording was too short or invalid. Please try again.")
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")

st.divider()
st.caption("Developed for Introduction to NLP @ Spring 2026. Model trained on STT4SG-350 corpus.")