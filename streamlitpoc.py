#!/usr/bin/env python
# coding: utf-8

import os
import datetime
from PIL import Image
import torchvision.transforms as T
import streamlit as st
import torch

# Charset helper
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

# Decoder
def ctc_greedy_decoder(logits, charset, blank_index=0):
    T, B, C = logits.shape
    preds = logits.argmax(dim=2).transpose(0,1)
    results = []
    for arr in preds:
        prev = None
        out_chars = []
        for p in arr.tolist():
            if p != prev and p != blank_index:
                out_chars.append(charset.idx2char.get(p, ""))
            prev = p
        results.append("".join(out_chars))
    return results

# Inference wrapper using TorchScript
class InferenceModel:
    def __init__(self, ts_path="best_model_ts.pt", device="cpu"):
        self.device = device
        self.charset = Charset()
        self.model = torch.jit.load(ts_path, map_location=device)
        self.model.eval()
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])
    def preprocess(self, pil_img, img_height=64, max_width=1600):
        w, h = pil_img.size
        new_h = img_height
        new_w = max(1, int(w * (new_h / float(h))))
        if new_w > max_width:
            new_w = max_width
        pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        tensor = self.transform(pil_img).unsqueeze(0)
        return tensor
    def predict(self, pil_img):
        x = self.preprocess(pil_img).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
        return ctc_greedy_decoder(logits.cpu(), self.charset)[0]

# Streamlit UI
st.set_page_config(page_title="Doctor Notes Digitizer", layout="wide")
st.title("🩺 Handwritten Notes Digitizer")

@st.cache_resource
def load_model():
    return InferenceModel(ts_path="best_model_ts.pt")

infr = load_model()

uploaded_file = st.file_uploader("Upload a picture of your handwritten note", type=["jpg", "jpeg", "png"])
if uploaded_file:
    pil = Image.open(uploaded_file)
    st.image(pil, caption="Uploaded Note", use_column_width=True)
    with st.spinner("Transcribing..."):
        pred = infr.predict(pil)
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
