import streamlit as st
import torch
import numpy as np
import io
import subprocess
import joblib
import gc
from transformers import AutoProcessor, Wav2Vec2Model
from streamlit_mic_recorder import mic_recorder

# --- PAGE CONFIG ---
st.set_page_config(page_title="ZHAW Dialect AI", page_icon="🇨🇭", layout="centered")

# ZHAW Branding
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #006494; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Lädt Wav2Vec2 in FP16 und das Baseline-Modell."""
    model_id = "facebook/wav2vec2-large-xlsr-53-german"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # FP16 spart 50% RAM auf Streamlit Cloud
    wav_model = Wav2Vec2Model.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).eval()
    
    # Lade die neue baseline_results.pkl
    data = joblib.load('baseline_results.pkl')
    
    # Extrahiere Modell und Klassenliste
    classifier = data['model']
    classes = data['classes']
    
    gc.collect()
    return processor, wav_model, classifier, classes

def load_audio_ffmpeg(audio_bytes, sr=16000):
    """Exakte Kopie deiner Kaggle-Logik für Audio-Daten."""
    cmd = [
        'ffmpeg', '-threads', '1', '-v', 'error', '-i', 'pipe:0', 
        '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', str(sr), '-ac', '1', '-'
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = proc.communicate(input=audio_bytes)
        if out:
            # Deine Kaggle-Normierung:
            return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return None
    except:
        return None

# --- UI ---
st.title("🇨🇭 Swiss German Age Prediction")
st.markdown("### Baseline Model: Full Age Group Classification")

try:
    processor, wav_model, classifier, AGE_CLASSES = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error: Make sure 'baseline_results.pkl' contains the 'model' key. ({e})")

st.divider()

st.subheader("Record Audio")
audio_data = mic_recorder(start_prompt="🔴 Start Recording", stop_prompt="⏹️ Stop Recording", key='recorder')

if audio_data:
    audio_bytes = audio_data['bytes']
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("🚀 Analyze Age Group"):
        with st.spinner('AI is processing (Kaggle Pipeline)...'):
            try:
                # 1. FFmpeg Loading
                aud = load_audio_ffmpeg(audio_bytes)
                
                if aud is not None and len(aud) > 1600:
                    # 2. Kaggle Preprocessing: max 5 seconds (80k samples)
                    aud = aud[:80000]
                    
                    # 3. Wav2Vec2 Embedding (FP16)
                    inputs = processor(aud, sampling_rate=16000, return_tensors="pt", padding=True)
                    inputs = {k: v.to(torch.float16) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = wav_model(**inputs)
                        # Mean Pooling (wie in Kaggle)
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
                    
                    # 4. Predict
                    pred_label = classifier.predict(emb)[0]
                    
                    st.divider()
                    st.balloons()
                    st.success(f"## Predicted Class: **{pred_label}**")
                    st.write(f"Recognized Classes: {', '.join(AGE_CLASSES)}")
                else:
                    st.warning("Recording too short.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

st.divider()
st.caption("Baseline Model | LGBMClassifier | Wav2Vec2-XLSR-53")