#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import streamlit as st


# In[8]:


import sys
import streamlit as st

st.write("Python executable:", sys.executable)
st.write("Python version:", sys.version)


# In[2]:


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


# In[3]:


# CRNN Model
class SmallCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), stride=(2,1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2,1), stride=(2,1)),
        )
        self.conv1x1 = nn.Conv2d(512, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1x1(x)
        return x

class CRNN(nn.Module):
    def __init__(self, num_classes, in_channels=1, hidden_size=256, num_layers=2):
        super().__init__()
        self.cnn = SmallCNN(in_channels=in_channels, out_channels=512)
        self.rnn = nn.LSTM(512, hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c2, h2, w2 = conv.size()
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        logits = self.fc(rnn_out)
        logits = logits.permute(1, 0, 2)  # [T,B,C]
        return logits


# In[4]:


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


# In[5]:


# Inference wrapper
class InferenceModel:
    def __init__(self, ckpt_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.charset = Charset()
        self.model = CRNN(num_classes=self.charset.num_classes, in_channels=1).to(self.device)
        if ckpt_path and os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location=self.device)
            if isinstance(state, dict) and "model_state" in state:
                self.model.load_state_dict(state["model_state"])
            else:
                self.model.load_state_dict(state)
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


# In[6]:


# Streamlit UI
st.set_page_config(page_title="Doctor Notes Digitizer", layout="wide")
st.title("🩺 Handwritten Notes Digitizer")

@st.cache_resource
def load_model():
    return InferenceModel(ckpt_path="best_model.pt")

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


# In[ ]:




