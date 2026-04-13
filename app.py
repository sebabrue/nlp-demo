import streamlit as st
from st_audiorec import st_audiorec

st.title("🎙️ Audio Recorder App")
st.write("Click the button below to record your voice.")

# This creates the recording widget
wav_audio_data = st_audiorec()

# If audio has been recorded, show the player
if wav_audio_data is not None:
    st.audio(wav_audio_data, format="audio/wav")
    st.success("Audio recorded successfully!")
