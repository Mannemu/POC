#!/usr/bin/env python
# coding: utf-8

import os
import datetime
from PIL import Image, ImageOps
import numpy as np
import streamlit as st
import torch
# load TorchScript model
import torch

model = torch.jit.load("best_model_ts.pt", map_location="cpu")
model.eval()


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

# Preprocessing function using Pillow + NumPy
def preprocess_image(pil_img, img_height=64, max_width=1600):
    w, h = pil_img.size
    new_h = img_height
    new_w = max(1, int(w * (new_h / float(h))))
    if new_w > max_width:
        new_w = max_width
    pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
    pil_img = ImageOps.grayscale(pil_img)
    tensor = np.array(pil_img, dtype=np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5  # normalize
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return tensor

# Inference wrapper using TorchScript
class InferenceModel:
    def __init__(self, ts_path="best_model_ts.pt", device="cpu"):
        self.device = device
        self.charset = Charset()
        self.model = torch.jit.load(ts_path, map_location=device)
        self.model.eval()
    def predict(self, pil_img):
        x = preprocess_image(pil_img).to(self.device)
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
    correc
