import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
#uzgarish
st.title("Speech-to-Text va Text-to-Speech")

# Speech-to-Text yuklash
st.header("Speech-to-Text")
if "speech_to_text_model.pkl" in os.listdir():
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    model.load_state_dict(torch.load("speech_to_text_model.pkl"))
    st.write("Speech-to-Text modeli yuklandi!")

# Text-to-Speech yuklash
st.header("Text-to-Speech")
if "text_to_speech_model.pkl" in os.listdir():
    with open("text_to_speech_model.pkl", "rb") as f:
        tts = pickle.load(f)
    st.write("Text-to-Speech modeli yuklandi!")
