#import libraries

import os
import datetime
import numpy as np
from PIL import Image
import streamlit as st
import torch

st.set_page_config(page_title="Doctor Notes Digitizer", layout="wide")
st.title("🩺 Handwritten Notes Digitizer")

# Load TorchScript model
@st.cache_resource
def load_model():
    if not os.path.exists("best_model_ts.pt"):
        st.error("Model file best_model_ts.pt not found!")
        st.stop()
    model = torch.jit.load("best_model_ts.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()

# Charset for decoding
class Charset:
    def __init__(self, chars="abcdefghijklmnopqrstuvwxyz0123456789.,;:!?/()-' \""):
        unique = []
        for c in chars:
            if c not in unique:
                unique.append(c)
        self.chars = unique
        self.idx2char = {i+1: c for i, c in enumerate(self.chars)}
        self.idx2char[0] = ""  # blank
        self.char2idx = {c: i+1 for i, c in enumerate(self.chars)}
        self.blank_idx = 0
        self.num_classes = len(self.chars) + 1

charset = Charset()
# Simple CTC Greedy Decoder
def ctc_greedy_decoder(logits, charset):
    preds = logits.argmax(axis=2)[0]  # assume batch=1
    prev = None
    out_chars = []
    for p in preds.tolist():
        if p != prev and p != charset.blank_idx:
            out_chars.append(charset.idx2char.get(p, ""))
        prev = p
    return "".join(out_chars)

# Preprocess image
def preprocess(pil_img, img_height=64, max_width=1600):
    w, h = pil_img.size
    new_h = img_height
    new_w = max(1, int(w * (new_h / float(h))))
    if new_w > max_width:
        new_w = max_width
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    img_arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    img_arr = (img_arr - 0.5) / 0.5  # normalize
    tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0)  # [B,C,H,W]
    return tensor


# File uploader and inference

uploaded_file = st.file_uploader("Upload a picture of your handwritten note", type=["jpg", "jpeg", "png"])

if uploaded_file:
    pil = Image.open(uploaded_file)
    st.image(pil, caption="Uploaded Note", use_column_width=True)

    with st.spinner("Transcribing..."):
        x = preprocess(pil)
        with torch.no_grad():
            logits = model(x)
        pred = ctc_greedy_decoder(logits.cpu(), charset)

    st.subheader("Transcribed Text")
    corrected_text = st.text_area("Correct the transcription if needed:", pred, height=200)
    feedback = st.text_area("Feedback for the developer (optional):", "")

    if st.button("Submit"):
        os.makedirs("feedback_data", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"feedback_data/feedback_{ts}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write("Raw Prediction:\n" + pred + "\n\n")
            f.write("Corrected:\n" + corrected_text + "\n\n")
            f.write("Feedback:\n" + feedback + "\n")
        st.success("Submitted — thank you!")
else:
    st.info("Please upload a handwritten note to begin.")
