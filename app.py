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
st.set_page_config(page_title="Swiss Dialect AI", page_icon="🇨🇭", layout="centered")

# ZHAW-style branding
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    stButton>button { width: 100%; border-radius: 8px; height: 3.5em; background-color: #006494; color: white; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIGURATION ---
# IMPORTANT: Adjust this list to match your exact LabelEncoder.classes_ from Kaggle
AGE_CLASSES = ["15-24", "25-34", "35-44", "45-54", "55-64", "65+"] 

@st.cache_resource
def load_models():
    """Loads models using Half-Precision (FP16) to save RAM."""
    model_id = "facebook/wav2vec2-large-xlsr-53-german"
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Load Wav2Vec2 in FP16 (Uses ~50% less RAM)
    wav_model = Wav2Vec2Model.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).eval()
    
    # Load your local classifier
    data = joblib.load('research_results.pkl') 
    if isinstance(data, dict):
        classifier = data.get('model', list(data.values())[0])
    else:
        classifier = data
        
    gc.collect() # Force garbage collection
    return processor, wav_model, classifier

def load_audio_from_bytes_ffmpeg(audio_bytes, sr=16000):
    """Kaggle-style FFmpeg loading: normalization to [-1, 1]."""
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
st.markdown("### Acoustic Analysis of Generational Dialect Change")

try:
    processor, wav_model, classifier = load_models()
except Exception as e:
    st.error(f"Error loading models: {e}")

st.divider()

st.subheader("Step 1: Record your voice")
st.write("Speak for about 5 seconds in your natural Swiss German dialect.")

audio_data = mic_recorder(
    start_prompt="🔴 Start Recording", 
    stop_prompt="⏹️ Stop Recording", 
    key='recorder'
)

if audio_data:
    audio_bytes = audio_data['bytes']
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("🚀 Run Precise Analysis"):
        with st.spinner('AI is analyzing your voice...'):
            try:
                # 1. Load via FFmpeg
                aud = load_audio_from_bytes_ffmpeg(audio_bytes)
                
                if aud is not None and len(aud) > 1600:
                    # 2. Preprocessing (matching Kaggle: 5s limit)
                    aud = aud[:80000]
                    
                    # 3. Feature Extraction with FP16
                    inputs = processor(aud, sampling_rate=16000, return_tensors="pt", padding=True)
                    # Convert inputs to float16 to match the model
                    inputs = {k: v.to(torch.float16) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = wav_model(**inputs)
                        # Mean Pooling and back to float32 for the classifier
                        emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
                    
                    # 4. Prediction
                    prediction_idx = classifier.predict(emb)[0]
                    
                    # 5. Mapping
                    if isinstance(prediction_idx, (int, np.integer)):
                        result_label = AGE_CLASSES[prediction_idx]
                    else:
                        result_label = prediction_idx

                    st.divider()
                    st.balloons()
                    st.success(f"## Predicted Age Group: **{result_label}**")
                    st.info("The prediction reflects the phonetic characteristics found in your age cohort.")
                else:
                    st.warning("Recording too short. Please speak longer.")
            
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.info("This is often a memory limit issue. Try to 'Reboot App' in the Streamlit menu.")

st.divider()
st.caption("Developed for Intro to NLP @ Spring 2026. Data: STT4SG-350.")