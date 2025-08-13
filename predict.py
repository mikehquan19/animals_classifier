import requests
from PIL import Image
# from matplotlib import pyplot as plt
from io import BytesIO
from torchvision.transforms import v2
import torch
from torch.nn import Module
from dataset import idx_to_label
from model.resnet import ResNet50Classifier


def predict(arg_model: Module, arg_img_url: str) -> str:
    arg_model.eval()

    # Load the image
    response = requests.get(arg_img_url)
    img = Image.open(BytesIO(response.content))

    # Convert all the images to RGB to guarantee shape (3, 224, 224)
    if img.mode != 'RGB': img = img.convert('RGB')

    # Transform the image to necessary form
    resize_transform = v2.Compose([
        v2.Resize([224, 224]),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            # the value of mean and stdev of normalization
            mean=[0.5204, 0.5028, 0.4156],
            std=[0.2667, 0.2621, 0.2797])
    ])
    img_tensor = resize_transform(img)

    # Show the image
    """
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.show()
    """

    img_tensor = img_tensor.unsqueeze(0).to(
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    _, predicted = torch.max(arg_model(img_tensor), dim=1)

    return idx_to_label[int(predicted.item())]


if __name__ == "__main__": 
    # initialize the model and load the pre-trained weight 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    animals_classifier = ResNet50Classifier().to(device=device)
    animals_classifier.load_state_dict(torch.load('./data/animals_checkpoint.tph', map_location=device)["model"])

    img_url = "https://wallup.net/wp-content/uploads/2018/10/06/364377-puppies-puppy-baby-dog-dogs-41.jpg"
    print(predict(animals_classifier, img_url))


