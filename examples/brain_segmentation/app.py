# This example uses the LGG Segmentation dataset.
# You can download the dataset from https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data.

import os

from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from PIL import Image
import torch
import glob
import numpy as np
import pandas as pd
from speedygui.predictor import Predictor

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


class BrainSegmentationDataset(Dataset):
    def __init__(self, folder_path):
        mask_paths = glob.glob(f'{folder_path}/*_mask.tif')

        masks_paths = [mask_path for mask_path in mask_paths]
        image_paths = [mask_path.replace('_mask', '') for mask_path in masks_paths]

        self.path_df = pd.DataFrame({'image': image_paths, 'mask': masks_paths})

        self.preprocess_image = v2.Compose([
            v2.PILToTensor(),
            v2.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            v2.ToDtype(torch.float32),
        ])

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, idx):
        epsilon = 1e-8

        # Load the image and mask from their paths
        image_path = self.path_df.loc[idx, 'image']
        mask_path = self.path_df.loc[idx, 'mask']

        image = Image.open(image_path).convert('RGB')
        m, s = np.mean(np.array(image), axis=(0, 1)), np.std(np.array(image), axis=(0, 1))
        # Add epsilon to std to avoid division by zero
        s = np.where(s == 0, epsilon, s)
        mask = Image.open(mask_path).convert('L')

        preprocess = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float32),
            v2.Normalize(mean=m.tolist(), std=s.tolist()),
        ])

        # Apply transformations
        image = preprocess(image)
        mask = preprocess(mask)

        return image, mask


def dataset_creator(folder_path):
    return BrainSegmentationDataset(folder_path)


segmentation_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                    in_channels=3, out_channels=1, init_features=32, pretrained=True)


def output_transform(outputs):
    return outputs[0]


def save_predictions_fn(folder_path, predictions, dataset):
    """
    Saves the predictions and returns a string to the path that the predictions were saved
    """
    predicted_masks_dir = os.path.join(folder_path, "predicted_masks")
    os.makedirs(predicted_masks_dir, exist_ok=True)

    for idx in range(len(dataset)):
        image_path = dataset.path_df.loc[idx, 'image']
        image_name = os.path.basename(image_path)
        mask_path = os.path.join(predicted_masks_dir, f"predicted_{image_name}")

        # Get the prediction for the current image
        pred = predictions[idx]
        binary_mask = (pred.cpu().numpy()).astype(np.uint8) * 255

        # Save image as tiff
        mask_image = Image.fromarray(binary_mask).convert("L")
        mask_image.save(mask_path, format="TIFF")

    return predicted_masks_dir


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = segmentation_model
dataloader_kwargs = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 0,
}
app_name = 'Predictor App'
formal_name = 'org.example.predictor'
description = ("To use this application, start by selecting a folder such as 'TCGA_CS_4941_19960909' containing tiff "
               "images and masks")


def main():
    predictor = Predictor(model, dataset_creator, dataloader_kwargs, save_predictions_fn=save_predictions_fn,
                          output_transform=output_transform, device=device)
    app = predictor.create_app(app_name, formal_name, description=description)
    return app
