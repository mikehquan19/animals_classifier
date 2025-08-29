import requests
from PIL import Image, ImageFile
# from matplotlib import pyplot as plt
from io import BytesIO
from torchvision.transforms import v2
import torch
from dataset import idx_to_english
from model import get_model
import argparse

@torch.no_grad()
def predict(arg_model: torch.nn.Module, arg_img: ImageFile) -> str:
    """ Predict the image using the model """
    arg_model.eval()

    # Convert all the images to RGB to guarantee shape (3, 224, 224)
    if arg_img.mode != 'RGB': arg_img = arg_img.convert('RGB')

    # Transform the image to necessary form
    resize_transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            # the value of mean and stdev of normalization
            mean=[0.5204, 0.5028, 0.4156],
            std=[0.2667, 0.2621, 0.2797]
        )
    ])

    img_tensor = resize_transform(arg_img).unsqueeze(0).to(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    _, predicted = torch.max(arg_model(img_tensor), dim=1)

    return idx_to_english[int(predicted.item())]


if __name__ == "__main__": 
    # For testing, initialize the model and load the pre-trained weight 
    parser = argparse.ArgumentParser(
        description="inference from the trained classification model")
    
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("img_url", type=str, help="The image url of the model")

    animals_classifier = get_model(
        parser.parse_args().model_name, load_state=True
    )
    # Load the image
    response = requests.get(parser.parse_args().img_url)
    img = Image.open(BytesIO(response.content))
    print(predict(animals_classifier, img))




