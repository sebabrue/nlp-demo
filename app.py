import io

import joblib
import librosa
import numpy as np
import streamlit as st
import torch
from streamlit_mic_recorder import mic_recorder
from transformers import AutoProcessor, Wav2Vec2Model


st.set_page_config(page_title="Swiss Dialect AI", page_icon="🇨🇭", layout="wide")

st.markdown(
    """
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #E67E22;
        color: white;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #d96411;
    }
    h1, h2, h3 {
        color: #156082;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🇨🇭 Swiss German Generation Predictor")
st.markdown(
    """
    Analyze Swiss German dialect features to predict speaker age group.
    This app uses acoustic embeddings and two trained classifiers.
    """
)


@st.cache_resource
def load_models():
    model_id = "facebook/wav2vec2-large-xlsr-53-german"
    processor = AutoProcessor.from_pretrained(model_id)
    wav_model = Wav2Vec2Model.from_pretrained(model_id).eval()

    baseline_data = joblib.load("baseline_results.pkl")
    research_data = joblib.load("research_results.pkl")

    baseline_model = baseline_data["model"]
    baseline_classes = baseline_data.get("classes", [])
    research_model = research_data["model"]

    return processor, wav_model, baseline_model, baseline_classes, research_model


with st.spinner("Loading AI models..."):
    try:
        processor, wav_model, baseline_model, baseline_classes, research_model = load_models()
    except Exception as exc:
        st.error(f"Error loading models: {exc}")
        st.stop()


st.divider()
left, right = st.columns([2, 1])
with left:
    st.subheader("Record your voice")
    st.info("Speak for about 5 to 10 seconds in your natural Swiss German dialect.")
with right:
    st.markdown("###")
    st.markdown("**Tip:** Speak naturally and clearly.")

audio_data = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    key="recorder",
)

if audio_data and audio_data.get("bytes"):
    audio_bytes = audio_data["bytes"]
    st.audio(audio_bytes, format="audio/wav")

    if st.button("🔍 Analyze Acoustic Features", use_container_width=True):
        with st.spinner("Extracting embeddings and running predictions..."):
            try:
                audio_file = io.BytesIO(audio_bytes)
                audio, _ = librosa.load(audio_file, sr=16000)

                if audio.size == 0:
                    st.error("The recorded audio is empty.")
                    st.stop()

                if len(audio) > 160000:
                    audio = audio[:160000]

                inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
                with torch.no_grad():
                    outputs = wav_model(**inputs)
                    features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                baseline_pred = baseline_model.predict(features)[0]
                baseline_prob = baseline_model.predict_proba(features)[0]
                research_pred = research_model.predict(features)[0]
                research_prob = research_model.predict_proba(features)[0]

                age_class_map = {
                    0: "Teens",
                    1: "Twenties",
                    2: "Thirties",
                    3: "Forties",
                    4: "Sixties",
                    5: "Seventies",
                }
                baseline_labels = [age_class_map.get(cls, f"Class {cls}") for cls in baseline_classes] if baseline_classes else [age_class_map.get(i, f"Class {i}") for i in range(len(baseline_prob))]
                binary_label_map = {0: "Young (Teens/Twenties)", 1: "Old (Sixties/Seventies)"}

                st.divider()
                col_left, col_right = st.columns(2)

                with col_left:
                    st.markdown("### Baseline model")
                    st.success(f"Predicted age class: {age_class_map.get(int(baseline_pred), f'Class {baseline_pred}')}")
                    baseline_probs = {label: round(float(prob) * 100, 1) for label, prob in zip(baseline_labels, baseline_prob)}
                    st.bar_chart(baseline_probs)

                with col_right:
                    st.markdown("### Research model")
                    st.success(f"Predicted group: {binary_label_map.get(int(research_pred), f'Class {research_pred}')}")
                    research_probs = {
                        "Young": round(float(research_prob[0]) * 100, 1),
                        "Old": round(float(research_prob[1]) * 100, 1),
                    }
                    st.bar_chart(research_probs)

                st.caption("Predictions are based on Wav2Vec2 acoustic embeddings and LightGBM classifiers.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")


st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #156082; margin-top: 30px;">
    <small>
    <strong>Project:</strong> Acoustic Fingerprints of Generational Change in Swiss German<br>
    <strong>Course:</strong> Introduction to NLP (Spring 2026)<br>
    <strong>Data:</strong> STT4SG-350 Corpus
    </small>
    </div>
    """,
    unsafe_allow_html=True,
)