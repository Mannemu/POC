import gradio as gr
from PIL import Image
import torch
import torchvision.transforms as T
import os

# --------- CHARSET & MODEL CLASSES ---------
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

class SmallCNN(torch.nn.Module):
    # ... copy your CNN definition here ...

class CRNN(torch.nn.Module):
    # ... copy your CRNN definition here ...

def ctc_greedy_decoder(logits, charset, blank_index=0):
    T_, B, C = logits.shape
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

class InferenceModel:
    def __init__(self, ckpt_path="best_model.pt", device=None):
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

inference = InferenceModel()

# --------- GRADIO INTERFACE ---------
def transcribe_image(image):
    if image is None:
        return "Upload an image"
    return inference.predict(image)

iface = gr.Interface(
    fn=transcribe_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="Transcribed Text"),
    live=False,
)
iface.launch()
