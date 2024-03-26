from typing import Dict, Callable, Optional, Union
import torch
from torch.utils.data import DataLoader, Dataset

from .predictor_app import PredictorApp


class Predictor:
    def __init__(self, model, dataset_creator, dataloader_kwargs: Dict, save_predictions_fn: Callable, progress_callback: Optional[Callable] = None, device: torch.device = torch.device('cpu'), output_transform: Optional[Callable] = None):
        self.model = model.to(device)
        self.device = device
        self.dataset_creator = dataset_creator
        self.dataloader_kwargs = dataloader_kwargs
        self.progress_callback = progress_callback
        self.save_predictions_fn = save_predictions_fn
        self.output_transform = output_transform
        self.example_inputs = []
        self.example_outputs = []

    def create_dataloaders(self, data: Dict[str, Union[Dataset, Dict]]) -> Dict[str, DataLoader]:
        """
        Creates dataloaders from the given data dictionary.

        Args:
            data (Dict[str, Union[Dataset, Dict]]): A dictionary containing the datasets or data dictionaries for each phase.

        Returns:
            Dict[str, DataLoader]: A dictionary of dataloaders for each phase.
        """
        dataloaders = {}
        for phase, dataset_or_data in data.items():
            if isinstance(dataset_or_data, Dataset):
                dataset = dataset_or_data
            else:
                dataset = self.dataset_creator(**dataset_or_data)
            dataloaders[phase] = DataLoader(dataset, **self.dataloader_kwargs)
        return dataloaders

    def predict(self, data: Dict[str, Union[Dataset, Dict]], batch_examples: int = 0) -> Dict[str, torch.Tensor]:
        """
        Performs prediction using the model and applies an optional transformation function to the outputs.

        Args:
            data (Dict[str, Union[Dataset, Dict]]): A dictionary containing the datasets or data dictionaries for each phase.
            batch_examples (int): The number of batches to store examples from. These examples include the input, mask, and predicted output.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys corresponding to phases and values being the transformed model outputs.
        """
        self.model.eval()
        results = {}
        dataloaders = self.create_dataloaders(data)

        with torch.no_grad():
            for phase, dataloader in dataloaders.items():
                all_predictions = []
                for batch_idx, (inputs, _) in enumerate(dataloader):
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    # Apply the optional transformation function if provided
                    if self.output_transform:
                        outputs = self.output_transform(outputs)
                    # Check if outputs is not a PyTorch tensor, convert if necessary
                    if not isinstance(outputs, torch.Tensor):
                        outputs = torch.tensor(outputs, device=self.device)
                    all_predictions.append(outputs.cpu().detach())

                    # Store examples if needed
                    if batch_examples > 0 and batch_idx < batch_examples:
                        self.example_inputs.extend(inputs.cpu().detach())
                        self.example_outputs.extend(outputs.cpu().detach())

                    # Update progress
                    if self.progress_callback:
                        progress = int((batch_idx + 1) / len(dataloader) * 100)
                        self.progress_callback(progress)

                # Convert list of tensors to a single tensor
                results[phase] = torch.cat(all_predictions, dim=0)

        return results

    def create_app(self, app_name, formal_name, mean=None, std=None, **kwargs):
        """
        Creates and returns the Toga application instance with the predictor functionality.
        """
        app = PredictorApp(app_name, formal_name, self, mean=mean, std=std, **kwargs)
        return app
