import streamlit as st
import torch
import numpy as np
import joblib
import subprocess
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Setup and Config
MODEL_ID = "facebook/wav2vec2-large-xlsr-53-german"
st.set_page_config(page_title="Voice Age Classifier", page_icon="🎙️", layout="centered")

st.markdown(
    """
    <style>
    .block-container {
        max-width: 760px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    @media (max-width: 640px) {
        .block-container {
            max-width: 100%;
            padding-top: 1rem;
        }

        .stButton > button {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    # Load processor and embedding model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval()

    # Load classifiers
    clf_all = joblib.load("baseline_results.pkl")
    clf_bin = joblib.load("research_results.pkl")

    # Ensure we have the model object if it was saved as a dict
    model_all = clf_all["model"] if isinstance(clf_all, dict) else clf_all
    model_bin = clf_bin["model"] if isinstance(clf_bin, dict) else clf_bin

    return processor, model, model_all, model_bin


# Initialize app
processor, embedding_model, clf_all, clf_bin = load_assets()


def get_embeddings(audio_bytes):
    # Process audio through FFmpeg pipe (Matches training preprocessing)
    cmd = [
        "ffmpeg",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-",
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, _ = proc.communicate(input=audio_bytes)

    if out:
        audio_np = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        # Prepare for Wav2Vec2 (80k samples = 5 seconds)
        inputs = processor(
            audio_np[:80000], sampling_rate=16000, return_tensors="pt"
        ).input_values
        with torch.no_grad():
            return embedding_model(inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return None


# --- Minimalist UI ---
st.title("Swiss Voice Age Classifier")

# Built-in audio recorder
audio_file = st.audio_input("Record your voice", sample_rate=16000)

if audio_file:
    recorded_audio = audio_file.getvalue()

    # Trigger analysis automatically or via a single button
    if st.button("Predict Age", use_container_width=True):
        emb = get_embeddings(recorded_audio)

        if emb is not None:
            # Run Predictions
            res_all = clf_all.predict(emb)[0]
            res_bin = clf_bin.predict(emb)[0]

            # Formatting results
            label_bin = "young" if res_bin == 0 else "old"

            st.markdown("---")
            col1, col2 = st.columns(2, gap="small")
            col1.metric("Baseline Model", f"{res_all}")
            col2.metric("Research Model (Binary)", label_bin)
        else:
            st.error("Audio processing failed.")
