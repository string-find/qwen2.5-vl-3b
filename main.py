from fastapi import FastAPI, Query, HTTPException
import requests
from io import BytesIO
from PIL import Image
import time
import logging
import traceback
from transformers import AutoModelForCausalLM
import torch

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
            _hf_model = AutoModelForCausalLM.from_pretrained(
                "vikhyatk/moondream2",
                revision="2025-01-09",
                trust_remote_code=True,
                # Uncomment to run on GPU if available
                device_map="auto" if torch.cuda.is_available() else None
            )
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

@app.get("/")
def read_root():
    return {"message": "API is live. Use the /moondream/* endpoints for Moondream capabilities via Hugging Face."}

@app.get("/health")
def health_check():
    """Health check endpoint to verify the server is running"""
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/moondream/caption")
def moondream_caption(image_url: str = Query(...)):
    """
    Generate a caption for an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to caption
    """
    try:
        logger.info(f"Caption request received for image: {image_url}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = load_image_from_url(image_url)
        
        # Encode the image
        logger.info("Encoding image...")
        encoded_image = model.encode_image(image)
        
        # Generate caption
        logger.info("Generating caption...")
        caption = model.generate_caption(encoded_image)
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
def moondream_query(image_url: str = Query(...), question: str = Query(...)):
    """
    Query about an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to query
    - question: Question to ask about the image
    """
    try:
        logger.info(f"Query request received for image: {image_url}, question: {question}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = load_image_from_url(image_url)
        
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
def moondream_detect(image_url: str = Query(...), object_name: str = Query(...)):
    """
    Detect objects in an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to analyze
    - object_name: Object to detect in the image
    """
    try:
        logger.info(f"Detect request received for image: {image_url}, object: {object_name}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = load_image_from_url(image_url)
        
        # Encode the image
        encoded_image = model.encode_image(image)
        
        # Detect objects
        detection_result = model.detect(encoded_image, object_name)
        logger.info(f"Objects detected: {detection_result}")
        
        return {"objects": detection_result}
    except HTTPException as e:
        raise e
    except AttributeError:
        logger.error("Detect function not available in Hugging Face model implementation")
        raise HTTPException(status_code=501, detail="Detection functionality not implemented in the Hugging Face model version")
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")

@app.get("/moondream/point")
def moondream_point(image_url: str = Query(...), object_name: str = Query(...)):
    """
    Point at objects in an image using Moondream.
    
    Parameters:
    - image_url: URL of the image to analyze
    - object_name: Object to point at in the image
    """
    try:
        logger.info(f"Point request received for image: {image_url}, object: {object_name}")
        # Load the model
        model = get_model()
        
        # Load and process the image
        image = load_image_from_url(image_url)
        
        # Encode the image
        encoded_image = model.encode_image(image)
        
        # Point at objects
        point_result = model.point(encoded_image, object_name)
        logger.info(f"Points found: {point_result}")
        
        return {"points": point_result}
    except HTTPException as e:
        raise e
    except AttributeError:
        logger.error("Point function not available in Hugging Face model implementation")
        raise HTTPException(status_code=501, detail="Point functionality not implemented in the Hugging Face model version")
    except Exception as e:
        logger.error(f"Error in point endpoint: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error pointing at objects: {str(e)}")
