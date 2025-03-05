from fastapi import FastAPI, Query, HTTPException, Body
import requests
from io import BytesIO
from PIL import Image
import time
import logging
import traceback
from transformers import AutoModelForCausalLM
import torch
import base64
import re
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Lazy loading of the model
_hf_model = None

def get_model():
    global _hf_model
    if _hf_model is None:
        logger.info("Loading Moondream model from Hugging Face for the first time...")
        try:
            # Check if CUDA is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")
            
            # Load the model without device_map
            _hf_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                trust_remote_code=True
            )
            
            # Move model to the appropriate device
            _hf_model = _hf_model.to(device)
            logger.info("Moondream model loaded successfully from Hugging Face")
        except Exception as e:
            logger.error(f"Error loading Moondream model: {e}")
            logger.error(traceback.format_exc())
            raise e
    return _hf_model

def load_image_from_url(url):
    logger.info(f"Loading image from URL: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for HTTP errors
        image = Image.open(BytesIO(response.content))
        logger.info(f"Image loaded successfully, size: {image.size}")
        return image
    except requests.exceptions.RequestException as e:
        logger.error(f"Error loading image from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Error loading image: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

def load_image_from_base64(base64_string):
    logger.info("Loading image from base64 string")
    try:
        # Remove data URL prefix if present
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
        
        # Decode base64 string
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        logger.info(f"Image loaded successfully from base64, size: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Error loading image from base64: {e}")
        raise HTTPException(status_code=400, detail=f"Error decoding base64 image: {str(e)}")

def get_image(image_url: Optional[str] = None, image_base64: Optional[str] = None):
    """Load image from URL or base64 string"""
    if image_url:
        return load_image_from_url(image_url)
    elif image_base64:
        return load_image_from_base64(image_base64)
    else:
        raise HTTPException(status_code=400, detail="Either image_url or image_base64 must be provided")

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /moondream/* endpoints for Moondream capabilities via Hugging Face."}

@app.get("/health")
def health_check():
    """Health check endpoint to verify the server is running"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/moondream/caption")
def moondream_caption(image_url: Optional[str] = Query(None), image_base64: Optional[str] = Query(None)):
    """
    Generate a caption for an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to caption (optional)
    - image_base64: Base64 encoded image data (optional)
    
    Either image_url or image_base64 must be provided.
    """
    try:
        logger.info(f"Caption request received")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = get_image(image_url, image_base64)
        
        # Encode the image
        logger.info("Encoding image...")
        encoded_image = model.encode_image(image)
        
        # Generate caption - try different method names
        logger.info("Generating caption...")
        try:
            # First try generate_caption
            caption = model.generate_caption(encoded_image)
        except AttributeError:
            # Fall back to caption method
            try:
                caption = model.caption(encoded_image)
            except AttributeError:
                # As a last resort, try general query with a caption prompt
                caption = model.query(encoded_image, "Describe this image briefly.")
                
        logger.info(f"Caption generated: {caption}")
        
        return {"caption": caption}
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error in caption endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")

@app.get("/moondream/query")
def moondream_query(
    question: str = Query(...),
    image_url: Optional[str] = Query(None),
    image_base64: Optional[str] = Query(None)
):
    """
    Query about an image using Moondream.
    
    Parameters:
    - question: Question to ask about the image
    - image_url: URL of the image to query (optional)
    - image_base64: Base64 encoded image data (optional)
    
    Either image_url or image_base64 must be provided.
    """
    try:
        logger.info(f"Query request received, question: {question}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = get_image(image_url, image_base64)
        
        # Encode the image
        encoded_image = model.encode_image(image)
        
        # Generate answer
        answer = model.query(encoded_image, question)
        logger.info(f"Query answered: {answer}")
        
        return {"answer": answer}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in query endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error answering query: {str(e)}")

@app.get("/moondream/detect")
def moondream_detect(
    object_name: str = Query(...),
    image_url: Optional[str] = Query(None),
    image_base64: Optional[str] = Query(None)
):
    """
    Detect objects in an image using Moondream.
    
    Parameters:
    - object_name: Object to detect in the image
    - image_url: URL of the image to analyze (optional)
    - image_base64: Base64 encoded image data (optional)
    
    Either image_url or image_base64 must be provided.
    """
    try:
        logger.info(f"Detect request received, object: {object_name}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = get_image(image_url, image_base64)
        
        # Encode the image
        encoded_image = model.encode_image(image)
        
        # Detect objects - try different method implementations
        try:
            # First try with dictionary result style
            detection_result = model.detect(encoded_image, object_name)
            if isinstance(detection_result, dict) and "objects" in detection_result:
                logger.info(f"Objects detected: {detection_result['objects']}")
                return {"objects": detection_result["objects"]}
            else:
                # Direct result format
                logger.info(f"Objects detected: {detection_result}")
                return {"objects": detection_result}
        except AttributeError:
            # Try a workaround using query with specific detect wording
            logger.warning("Direct detect method not found, falling back to query-based detection")
            result = model.query(encoded_image, f"Please detect and list all {object_name} in this image with their positions.")
            return {"objects": [{"info": result}], "note": "Used query-based detection fallback"}
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")

@app.get("/moondream/point")
def moondream_point(
    object_name: str = Query(...),
    image_url: Optional[str] = Query(None),
    image_base64: Optional[str] = Query(None)
):
    """
    Point at objects in an image using Moondream.
    
    Parameters:
    - object_name: Object to point at in the image
    - image_url: URL of the image to analyze (optional)
    - image_base64: Base64 encoded image data (optional)
    
    Either image_url or image_base64 must be provided.
    """
    try:
        logger.info(f"Point request received, object: {object_name}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = get_image(image_url, image_base64)
        
        # Encode the image
        encoded_image = model.encode_image(image)
        
        # Point at objects - try different method implementations
        try:
            # First try with dictionary result style
            point_result = model.point(encoded_image, object_name)
            if isinstance(point_result, dict) and "points" in point_result:
                logger.info(f"Points found: {point_result['points']}")
                return {"points": point_result["points"]}
            else:
                # Direct result format
                logger.info(f"Points found: {point_result}")
                return {"points": point_result}
        except AttributeError:
            # Try a workaround using query with specific pointing wording
            logger.warning("Direct point method not found, falling back to query-based pointing")
            result = model.query(encoded_image, f"Please point out the coordinates of {object_name} in this image, giving x and y normalized coordinates.")
            return {"points": [{"info": result}], "note": "Used query-based pointing fallback"}
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in point endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error pointing at objects: {str(e)}")

# Post versions of the same endpoints to better handle large base64 payloads
@app.post("/moondream/caption")
def moondream_caption_post(
    image_url: Optional[str] = Body(None),
    image_base64: Optional[str] = Body(None)
):
    return moondream_caption(image_url, image_base64)

@app.post("/moondream/query")
def moondream_query_post(
    question: str = Body(...),
    image_url: Optional[str] = Body(None),
    image_base64: Optional[str] = Body(None)
):
    return moondream_query(question, image_url, image_base64)

@app.post("/moondream/detect")
def moondream_detect_post(
    object_name: str = Body(...),
    image_url: Optional[str] = Body(None),
    image_base64: Optional[str] = Body(None)
):
    return moondream_detect(object_name, image_url, image_base64)

@app.post("/moondream/point")
def moondream_point_post(
    object_name: str = Body(...),
    image_url: Optional[str] = Body(None),
    image_base64: Optional[str] = Body(None)
):
    return moondream_point(object_name, image_url, image_base64)
