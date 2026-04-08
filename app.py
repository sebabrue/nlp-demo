import streamlit as st
import torch
import numpy as np
import io
import librosa
import joblib
from transformers import AutoProcessor, Wav2Vec2Model
from streamlit_mic_recorder import mic_recorder

# --- PAGE CONFIG ---
st.set_page_config(page_title="Swiss Dialect AI", page_icon="🇨🇭", layout="centered")

# Custom CSS for branding
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #3498db;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🇨🇭 Swiss German Generation Predictor")
st.markdown("""
    This AI estimates whether a speaker belongs to the **Younger Generation** (Teens/20s) 
    or the **Older Generation** (60s/70s) based on acoustic dialect features.
""")

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    model_id = "facebook/wav2vec2-large-xlsr-53-german"
    processor = AutoProcessor.from_pretrained(model_id)
    wav_model = Wav2Vec2Model.from_pretrained(model_id).eval()
    
    # Load your trained model
    data = joblib.load('research_results.pkl') 
    
    if isinstance(data, dict) and 'model' in data:
        classifier = data['model']
    else:
        classifier = data
    return processor, wav_model, classifier

with st.spinner('Loading AI Models... Please wait.'):
    try:
        processor, wav_model, classifier = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")

# --- INTERFACE ---
st.subheader("Step 1: Record your voice")
st.info("Please speak for about 5 seconds in your natural Swiss German dialect.")

# Recording component
audio_data = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key='recorder'
)

if audio_data:
    audio_bytes = audio_data['bytes']
    st.audio(audio_bytes, format='audio/wav')
    
    if st.button("Analyze Acoustic Features"):
        with st.spinner('Extracting Wav2Vec2 Embeddings...'):
            try:
                # Process audio
                audio_file = io.BytesIO(audio_bytes)
                audio, sr = librosa.load(audio_file, sr=16000)
                
                # Cap at 5 seconds (matching training data)
                if len(audio) > 80000:
                    audio = audio[:80000]
                
                # Feature Extraction
                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = wav_model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1).numpy()
                
                # Prediction
                prediction = classifier.predict(features)[0] 
                
                # Output Mapping (Assuming 0=Young, 1=Old)
                st.divider()
                if prediction == 0 or str(prediction).lower() == 'jung':
                    st.balloons()
                    st.success("### Prediction: **Younger Generation** (Teens/20s)")
                else:
                    st.snow()
                    st.success("### Prediction: **Older Generation** (60s/70s+)")
                
                st.caption("Note: Prediction is based on spectral features and dialectal phonology.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")

# --- FOOTER ---
st.divider()
st.markdown("""
    **Project:** Acoustic Fingerprints of Generational Change  
    **Course:** Introduction to NLP (Spring 2026)  
    **Data:** STT4SG-350 Corpus
""")