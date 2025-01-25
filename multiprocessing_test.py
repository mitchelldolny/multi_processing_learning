

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import multiprocessing
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import pandas as pd
import time

def normalize(x):
    min_v = x.min(axis=0).values
    max_v = x.max(axis=0).values
    return (x - min_v) / (max_v - min_v), min_v, max_v

# A Customer dataset class
class getDataSet(Dataset):
    def __init__(self, images_path, csv_data, transform=None):
        self.images_path = images_path
        self.csv_data = csv_data
        self.transform = transform
        self.data, self.images, self.targets = self._load_data()
        
    def __len__(self):
        return len(self.data)

    def _load_data(self):
       
        images = []
        targets = self.csv_data[:, -6:]
        targets = np.log10(targets)  # Log transformation for targets
        targets = torch.tensor(targets, dtype=torch.float32)

        # Tabular data
        data = self.csv_data[:, 1:-6]
        data = scaler.fit_transform(data)  # Assuming 'scaler' is defined elsewhere
        ids = self.csv_data[:, 0]  # Image IDs

        # Create a manager to share data between processes
        manager = multiprocessing.Manager()
        images_list = manager.list()
        start_time = time.time()
        with multiprocessing.Pool() as pool:
            pool.starmap(self._load_image, [(img_id, images_list) for img_id in tqdm(ids)])
        end_time = time.time()
        print(f"Multiprocessing image loading time: {end_time - start_time:.2f} seconds")

        # Convert images from the manager list back to a regular list
        images = list(images_list)
        

        return torch.tensor(data, dtype=torch.float32), images, targets

    def _load_image(self, img_id, images_list):
        img_name = os.path.join(self.images_path, f"{int(img_id)}.jpeg")
        try:
            img = Image.open(img_name).convert('RGB')
            img = self.transform(img)
            images_list.append(img)
        except Exception as e:
            print("chicken")


    # Gets the item
    def __getitem__(self, idx):
        return self.data[idx], self.images[idx], self.targ


class getDataSetOld(Dataset):
    def __init__(self, images_path, csv_data, transform=None):
        self.images_path = images_path
        self.csv_data = csv_data
        self.transform = transform
        self.data, self.images, self.targets = self._load_data()
        
    def __len__(self):
        return len(self.data)

    def _load_data(self):
       
        images = []
        targets = self.csv_data[:, -6:]
        # Transform the targets
        targets = np.log10(targets)
        # Convert to a tensor
        targets = torch.tensor(targets, dtype=torch.float32)

        # Take the tabular data
        data = self.csv_data[:, 1:-6]
        # Scale it using the scalar 
        data = scaler.fit_transform(data)
        # Get the ids to use for finding pictures in order
        ids = self.csv_data[:,0]

        for i in tqdm(ids):
            # Find the picture
            img_name = os.path.join(self.images_path, f"{int(i)}.jpeg")
            # Transform it and convert to rgb (3 channels)
            img = self.transform(Image.open(img_name).convert('RGB'))
            # Append to images
            images.append(img)
        
        # Return 3 things the tabular data, images, and target values
        return torch.tensor(data, dtype=torch.float32), images, targets
    
    # Gets the item
    def __getitem__(self, idx):
        return self.data[idx], self.images[idx], self.targets[idx]



if __name__ == '__main__':
    directory_path = 'data/train_images'

    # Transformations for my training set
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(30),  # Randomly rotate image by up to 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Random color jitter
        transforms.RandomAffine(20, translate=(0.1, 0.1)),  # Random affine transformations (rotation, translation)
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the images
    ])

    # Load dataset
    train = getDataSet(directory_path, pd.read_csv('data/train.csv').values, transform=transform_train)

    train_old = getDataSetOld(directory_path, pd.read_csv('data/train.csv').values, transform=transform_train)

