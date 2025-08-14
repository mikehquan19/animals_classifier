from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
import requests
from PIL import Image
import io
import torch
from model.resnet import ResNet50Classifier
from predict import predict

# Intialize the app
app = FastAPI()

# Initialize the model and load weights
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
animals_classifier = ResNet50Classifier().to(device=device)
animals_classifier.load_state_dict(
    # Train the model first
    torch.load('./data/animals_checkpoint.pth', map_location=device)["model"])

class UploadRequest(BaseModel): 
    arg_url: str

@app.post("/upload_url")
async def upload_url(request: UploadRequest): 
    """ Upload the image through the URL then predict the animal """
    try: 
        response = requests.get(request.arg_url)
        image = Image.open(io.BytesIO(response.content))
    except:
        raise HTTPException(400, { 
            "status": 400, 
            "message": "Image can't be opened because of invalid URL."
        })
    animal_prediction = predict(animals_classifier, image)

    return {
        "status": 200, 
        "data": {
            "image_url": request.arg_url,
            "format": image.format,
            "result": animal_prediction
        }
    }


@app.post("/upload_image")
async def upload_image(arg_image: UploadFile): 
    """ 
    Upload the image file then predict the animal. 
    Only files with PNG, JPG, JPEG, or WEBP extension are accepted.
    """
    try: 
        image_content = await arg_image.read()
        image = Image.open(io.BytesIO(image_content))
    except: 
        raise HTTPException(400, { 
            "status": 400, 
            "message": "Image can't be opened because of invalid file."
        })
    animal_prediction = predict(animals_classifier, image)
    
    return {
        "status": 200, 
        "data": {
            "filename": arg_image.filename,
            "content_type": arg_image.content_type,
            "result": animal_prediction
        }
    }