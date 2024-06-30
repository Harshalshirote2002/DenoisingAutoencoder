from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
        self.__noise_factor = 0.5

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        noisy_image = self.add_noise(image)
        return noisy_image, image
    
    def add_noise(self, img):
        noisy_img = img + self.__noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape)
        return noisy_img
    
    def setNoiseFactor(self, factor):
        if factor < 0 or factor > 1:
            raise ValueError(f"Noise Factor must be in range: 0 to 1")
        self.__noise_factor = factor
        print(f"Noise Factor set to {factor}")
