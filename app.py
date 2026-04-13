import streamlit as st
from st_audiorec import st_audiorec

# --- UI Configuration ---
st.set_page_config(page_title="Swiss German Age Predictor", page_icon="🎤")

# Custom CSS for your hex colors: #156082 and #E67E22
st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #ffffff;
    }}
    h1 {{
        color: #156082;
    }}
    .stButton>button {{
        background-color: #E67E22;
        color: white;
        border-radius: 5px;
    }}
    </style>
    """,
    unsafe_allow_headers=True,
)

## Swiss German Age Classifier

st.write("Record a snippet of Swiss German audio to predict the speaker's age class.")

# --- Audio Recording Section ---
st.subheader("1. Record Audio")
wav_audio_data = st_audiorec()

# --- Placeholder for Future Prediction Logic ---
if wav_audio_data is not None:
    st.audio(wav_audio_data, format="audio/wav")
    st.success("Audio captured successfully!")

    if st.button("Predict Age Class"):
        st.info("Prediction model integration coming in the next step.")
        # This is where we will add:
        # 1. Feature extraction (e.g., Mel-spectrograms)
        # 2. Model inference (model.predict)
        # 3. Results display
else:
    st.warning("Please record your voice to enable prediction.")
