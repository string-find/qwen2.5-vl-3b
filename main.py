from fastapi import FastAPI, Query
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import requests
from io import BytesIO
from PIL import Image
import moondream as md

app = FastAPI()

# Qwen model setup
checkpoint = "Qwen/Qwen2.5-VL-7B-Instruct"
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained(
    checkpoint,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    # attn_implementation="flash_attention_2",
)

# Initialize Moondream model
moondream_model = md.vl(model='./moondream-2b-int8.mf.gz')

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /predict endpoint for Qwen or /moondream/* endpoints for Moondream."}

@app.get("/predict")
def predict(image_url: str = Query(...), prompt: str = Query(...)):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "image", "image": image_url}, {"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return {"response": output_texts[0]}

@app.get("/moondream/caption")
def moondream_caption(image_url: str = Query(...), length: str = Query("normal")):
    """
    Generate a caption for an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to caption
    - length: Length of the caption ("short" or "normal")
    """
    try:
        image = load_image_from_url(image_url)
        encoded_image = moondream_model.encode_image(image)
        result = moondream_model.caption(encoded_image, length=length)
        return {"caption": result["caption"]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/moondream/query")
def moondream_query(image_url: str = Query(...), question: str = Query(...)):
    """
    Query about an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to query
    - question: Question to ask about the image
    """
    try:
        image = load_image_from_url(image_url)
        encoded_image = moondream_model.encode_image(image)
        result = moondream_model.query(encoded_image, question)
        return {"answer": result["answer"]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/moondream/detect")
def moondream_detect(image_url: str = Query(...), object_name: str = Query(...)):
    """
    Detect objects in an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to analyze
    - object_name: Object to detect in the image
    """
    try:
        image = load_image_from_url(image_url)
        encoded_image = moondream_model.encode_image(image)
        result = moondream_model.detect(encoded_image, object_name)
        return {"objects": result["objects"]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/moondream/point")
def moondream_point(image_url: str = Query(...), object_name: str = Query(...)):
    """
    Point at objects in an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to analyze
    - object_name: Object to point at in the image
    """
    try:
        image = load_image_from_url(image_url)
        encoded_image = moondream_model.encode_image(image)
        result = moondream_model.point(encoded_image, object_name)
        return {"points": result["points"]}
    except Exception as e:
        return {"error": str(e)}
