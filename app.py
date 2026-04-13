import streamlit as st

st.title("My Ultra Basic App")
st.write("Hello world! I just deployed this.")

name = st.text_input("What's your name?")
if name:
    st.write(f"Nice to meet you, {name}!")
