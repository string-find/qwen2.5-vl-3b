import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from io import BytesIO
import os

import numpy as np
import torch
from PIL import Image
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span

app = FastAPI(title="Grounding DINO API")

model = None

class ImageRequest(BaseModel):
    image_url: str
    text_prompt: str
    box_threshold: float = 0.3
    text_threshold: float = 0.25

def load_image_from_url(image_url: str):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image_pil = Image.open(BytesIO(response.content)).convert("RGB")
        
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image, _ = transform(image_pil, None)
        return image_pil, image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")

def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, cpu_only=False):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption = caption + "."
    
    device = "cuda" if torch.cuda.is_available() and not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    
    logits = outputs["pred_logits"].sigmoid()[0]
    boxes = outputs["pred_boxes"][0]

    # Filter output
    logits_filt = logits.cpu().clone()
    boxes_filt = boxes.cpu().clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]
    boxes_filt = boxes_filt[filt_mask]

    # Get phrases
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
        pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")

    return boxes_filt, pred_phrases

@app.on_event("startup")
async def startup_event():
    global model
    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    print("Model loaded successfully")

@app.post("/predict")
async def predict(request: ImageRequest):
    try:
        # Load image from URL
        image_pil, image = load_image_from_url(request.image_url)
        
        # Get predictions
        boxes_filt, pred_phrases = get_grounding_output(
            model,
            image,
            request.text_prompt,
            request.box_threshold,
            request.text_threshold
        )

        # Prepare response
        response = {
            "boxes": boxes_filt.tolist(),
            "labels": pred_phrases,
            "image_size": [image_pil.height, image_pil.width]
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Original argparse code can be kept for CLI usage
    parser = argparse.ArgumentParser("Grounding DINO API", add_help=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)