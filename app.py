import streamlit as st
from st_audiorec import st_audiorec
import torch
import numpy as np
import joblib
import io
import subprocess
import logging

# 1. Dynamischer Import, um lokale Versionskonflikte zu umgehen
try:
    from transformers import AutoProcessor, Wav2Vec2Model
except ImportError:
    try:
        from transformers import Wav2Vec2Processor as AutoProcessor, Wav2Vec2Model
    except ImportError:
        st.error(
            "Kritischer Fehler: 'transformers' Bibliothek nicht gefunden oder zu alt."
        )
        st.stop()

# Warnungen unterdrücken
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# Konfiguration
MODEL_ID = "facebook/wav2vec2-large-xlsr-53-german"


# --- MODELL LADEN ---
@st.cache_resource
def load_models():
    # Lädt den Processor und das Embedding-Modell
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    model.eval()

    # Deine .pkl Dateien laden (müssen im gleichen Verzeichnis liegen)
    baseline_data = joblib.load("baseline_results.pkl")
    research_data = joblib.load("research_results.pkl")

    # Falls die .pkl Dateien ein Dictionary sind, extrahieren wir das Modell-Objekt
    clf_all = (
        baseline_data["model"] if isinstance(baseline_data, dict) else baseline_data
    )
    clf_bin = (
        research_data["model"] if isinstance(research_data, dict) else research_data
    )

    return processor, model, clf_all, clf_bin


# Komponenten initialisieren
try:
    processor, embedding_model, clf_all, clf_bin = load_models()
except Exception as e:
    st.error(f"Fehler beim Laden der Modelle/Dateien: {e}")
    st.stop()


# --- AUDIO VERARBEITUNG (FFmpeg) ---
def process_audio(audio_bytes):
    # Exakt wie in deinem Training: 16kHz, Mono, pcm_s16le
    cmd = [
        "ffmpeg",
        "-threads",
        "1",
        "-v",
        "error",
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
    try:
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out, err = proc.communicate(input=audio_bytes)
        if out:
            # Normalisierung auf float32
            return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
        return None
    except Exception as e:
        st.error(f"FFmpeg Fehler: {e}")
        return None


# --- UI ---
st.set_page_config(page_title="Age Predictor", page_icon="🎙️")
st.title("🎙️ Schweizerdeutsch Alters-Check")
st.write("Nimm ca. 5 Sekunden Audio auf, um dein Alter schätzen zu lassen.")

# Aufnahme-Widget
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    st.audio(wav_audio_data, format="audio/wav")

    if st.button("Stimme analysieren"):
        with st.spinner("Verarbeite Audio und generiere Embeddings..."):
            # 1. Audio via FFmpeg laden
            audio_np = process_audio(wav_audio_data)

            if audio_np is not None:
                # 2. Wav2Vec2 Embeddings (80.000 Samples = 5 Sek)
                inputs = processor(
                    audio_np[:80000], sampling_rate=16000, return_tensors="pt"
                ).input_values

                with torch.no_grad():
                    outputs = embedding_model(inputs)
                    # Mean Pooling über die Zeitachse (dim=1)
                    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                # 3. Predictions mit LightGBM
                pred_all = clf_all.predict(embeddings)[0]
                pred_bin_idx = clf_bin.predict(embeddings)[0]

                # Mapping für Binär-Label
                label_bin = (
                    "Jung (Teens/Twenties)" if pred_bin_idx == 0 else "Alt (60s+)"
                )

                # 4. Ergebnisse anzeigen
                st.divider()
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Baseline")
                    st.metric("Altersgruppe", f"{pred_all}")

                with col2:
                    st.subheader("Research")
                    st.metric("Tendenz", label_bin)
            else:
                st.error("Audio konnte nicht verarbeitet werden.")
