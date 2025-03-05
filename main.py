from fastapi import FastAPI, Query
import requests
from io import BytesIO
from PIL import Image
import moondream as md

app = FastAPI()

# Initialize Moondream model
moondream_model = md.vl(model='./moondream-2b-int8.mf.gz')

def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /moondream/* endpoints for Moondream capabilities."}

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
