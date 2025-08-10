import requests
from PIL import Image
from matplotlib import pyplot as plt
from io import BytesIO
from torchvision.transforms import v2
import torch
from torch.nn import Module
from dataset import idx_to_label
from model.resnet import ResNet101Classifier


def predict(arg_model: Module, arg_img_url: str) -> str:
    arg_model.eval()
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
            mean=[0.5200, 0.5028, 0.4161],
            std=[0.2667, 0.2623, 0.2798]
        )
    ])

    # Show the image
    img_tensor = resize_transform(img)
    plt.imshow(img_tensor.permute(1, 2, 0))
    plt.show()

    img_tensor = img_tensor.unsqueeze(0).to(torch.device('cuda'))
    _, predicted = torch.max(arg_model(img_tensor), dim=1)

    return idx_to_label[int(predicted.item())]


if __name__ == "__main__": 
    # initialize the model 
    classifier = ResNet101Classifier()
    classifier.to(torch.device('cuda'))
    # load the pre-trained weight the model 
    classifier.load_state_dict(torch.load('./animals_weight2', weights_only=True))

    img_url = "https://i.natgeofe.com/n/e0e24f3a-cef0-4499-b8d1-cdef05e6c4f4/NationalGeographic_1418626_3x2.jpg"
    print(predict(classifier, img_url))


