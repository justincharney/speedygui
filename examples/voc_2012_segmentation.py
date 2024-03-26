# This example uses the PASCAL VOC 2012 dataset.
# You can download the dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#data.

from skimage import color
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2
from PIL import Image
import torch
import os
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

from speedygui import Predictor


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return the image tensor and a dummy tensor (since we don't have targets)
        return image, torch.tensor(0)


def dataset_creator(folder_path, transform=None):
    image_paths = get_image_paths(folder_path, n=10)

    if not transform:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    dataset = SegmentationDataset(image_paths, transform=transform)
    return dataset


segmentation_model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)


def get_image_paths(folder_path, n=None):
    # This expects that we will select the parent folder with the JPEGImages folder as a child
    image_dir = os.path.join(folder_path, "JPEGImages")
    all_image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png"))
    ]

    if n is not None:
        return all_image_paths[:n]
    else:
        return all_image_paths


def output_transform(outputs):
    segmentation_masks = outputs["out"].argmax(dim=1)
    return segmentation_masks.cpu().numpy()


def save_predictions_fn(folder_path, predictions, dataset):
    predicted_masks_dir = os.path.join(folder_path, "predicted_masks")
    os.makedirs(predicted_masks_dir, exist_ok=True)

    for idx, pred in enumerate(predictions):
        image_name = os.path.basename(dataset.image_paths[idx])
        mask_path = os.path.join(predicted_masks_dir, f"predicted_{image_name}")
        color_pred = color.label2rgb(pred.cpu().numpy(), bg_label=0)
        mask_image = Image.fromarray((color_pred * 255).astype("uint8"))
        mask_image.save(mask_path)

    return predicted_masks_dir


# Setup
model = segmentation_model
dataloader_kwargs = {
    "batch_size": 1,
    "shuffle": False,
    "num_workers": 0,
}
app_name = 'Predictor App'
formal_name = 'org.example.predictor'
description = "To use this application, start by selecting the folder 'VOC2012'"

if __name__ == '__main__':
    predictor = Predictor(model, dataset_creator, dataloader_kwargs, save_predictions_fn=save_predictions_fn,
                          output_transform=output_transform)
    app = predictor.create_app(app_name, formal_name, description=description, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    app.main_loop()