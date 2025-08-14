import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path

label_to_idx = {"cane": 0, "cavallo": 1, "elefante": 2, "farfalla": 3, "gallina": 4, "gatto": 5, "mucca": 6, "pecora": 7, "ragno": 8, "scoiattolo": 9}
idx_to_label = {0: "dog", 1: "horse", 2: "elephant", 3: "butterfly", 4: "hen", 5: "cat", 6: "cow", 7: "sheep", 8: "spider", 9: "squirrel"}

class AnimalImages(Dataset):
    """ Dataset of the animal's images """

    def __init__(self, img_path: str, img_size: int, train: bool=True):
        # Verify if the path to directory exsists
        image_path = Path(img_path)
        if not image_path.is_dir(): raise Exception(f"{img_path} doesn't exist.")

        self.img_path = img_path
        self.img_size = img_size
        self.train = train
        self.data = []

        # create all the path to files
        directory_list = glob.glob(f"{self.img_path}/*") 
        for class_path in directory_list:

            # get label, last word of directory path
            label = class_path.split("/")[-1] 

            # split the list depending on whether its train set or test set
            img_path_list = []
            for img_extension in ["jpeg", "png", "jpg"]:
                img_path_list.extend(glob.glob(f"{class_path}/*.{img_extension}"))

            split = round(0.85 * len(img_path_list))
            img_path_list = img_path_list[:split] if train else img_path_list[split:]

            for img_path in img_path_list:
                # the tuple of path to the image and label of that image
                self.data.append([img_path, label])

        if self.train: random.shuffle(self.data)

        # image transform that applies augmentation 
        self.augmentation_transform = v2.Compose([
            v2.Resize((self.img_size, self.img_size)),
            v2.RandomResizedCrop(
                size=self.img_size, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
            v2.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.1),
            v2.RandomHorizontalFlip(),
            v2.RandomRotation(45), # apply rotation from -45 degrees to 45 degrees
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                # The value of mean and stdev of normalization
                mean=[0.5204, 0.5028, 0.4156],
                std=[0.2667, 0.2621, 0.2797])
        ])

        # image transform that doesn't apply augmentation
        self.no_augmentation_transform = v2.Compose([
            v2.Resize((self.img_size, self.img_size)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                # The value of mean and stdev of normalization
                mean=[0.5204, 0.5028, 0.4156],
                std=[0.2667, 0.2621, 0.2797]
            )
        ])


    def __len__(self):
        """ Total number of images of the dataset """
        return len(self.data)


    def __getitem__(self, index: int):
        """ Get the image of the dataset """

        # Open the image given the path
        img_path, label = self.data[index]
        img = Image.open(img_path)

        # Convert all the images to RGB to guarantee shape (3, 224, 224)
        if img.mode != "RGB": img = img.convert("RGB")

        # Resize, randomly augment the data, and tensorize them
        # depending on the type of dataset
        resize_transform = self.augmentation_transform if self.train else self.no_augmentation_transform

        # Process the images and the labels
        processed_img = resize_transform(img)
        label_idx = label_to_idx[label]
        return processed_img, label_idx