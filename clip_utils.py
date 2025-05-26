import cv2
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def load_clip_model(model_name="openai/clip-vit-base-patch16"):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    return model, processor

def preprocess_frame(frame):
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

def compute_probabilities(model, processor, image_pil, class_names, device):
    inputs = processor(text=class_names, images=image_pil, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    return probs
