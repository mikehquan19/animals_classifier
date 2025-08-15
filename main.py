from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
import io
import torch
from model import get_model
from predict import predict

# Intialize the app and the model with loaded weights
app = FastAPI()
animals_classifier = get_model('resnet50', './data/animals_checkpoint.pth')

class UploadRequest(BaseModel): 
    image_url: str

@app.post("/upload_image_url")
async def upload_url(request: UploadRequest): 
    """ Upload the image through the URL then predict the animal """
    try: 
        response = requests.get(request.image_url)
        image = Image.open(io.BytesIO(response.content))
    except:
        raise HTTPException(400, { 
            "status": 400, 
            "message": "Image can't be opened, possibly because of invalid URL."
        })
    animal_prediction = predict(animals_classifier, image)

    return {
        "status": 200, 
        "data": {
            "image_url": request.image_url,
            "format": image.format,
            "result": animal_prediction
        }
    }


@app.post("/upload_image_file")
async def upload_image(image_file: UploadFile): 
    """ 
    Upload the image file then predict the animal. 
    Only files with PNG, JPG, JPEG, or WEBP extension are accepted.
    """
    try: 
        image_content = await image_file.read()
        image = Image.open(io.BytesIO(image_content))
    except: 
        raise HTTPException(400, { 
            "status": 400, 
            "message": "Image can't be opened, possibly because of invalid file."
        })
    animal_prediction = predict(animals_classifier, image)
    
    return {
        "status": 200, 
        "data": {
            "filename": image_file.filename,
            "content_type": image_file.content_type,
            "result": animal_prediction
        }
    }