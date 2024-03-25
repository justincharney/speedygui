# Speedy GUI

Speedy GUI is a Python package that provides a quick template graphical user interface (GUI) for running inference on image segmentation datasets. 
It allows you to select a folder containing images, run a pre-trained PyTorch model, and save the predicted masks to a specified directory.

https://github.com/justincharney/speedygui/assets/56408885/6a6435fe-3073-4618-a31a-3188ea58eb25

## Installation

You can install the package from PyPI using pip:

```python
pip install speedygui
```

## Usage

To use the Predictor GUI, you need to provide the following components:

- A PyTorch model for prediction
- A dataset creator function to make a PyTorch Dataset
- Dataloader keyword arguments
- A save predictions function to store the outputs
- An optional output transformation function

Here's a basic example of how to create and run the Predictor GUI application:

```python
from speedygui import Predictor
import torch

# Define your dataset creator function
def dataset_creator(folder_path):
    # ...

# Load your model
model = torch.hub.load('your/model/path', 'model_name')

# Define dataloader keyword arguments
dataloader_kwargs = {
    'batch_size': 1,
    'shuffle': False,
    'num_workers': 0,
}

# Define a save predictions function
def save_predictions_fn(folder_path, predictions, dataset):
    # ...

# Define an optional output transform function
def output_transform(outputs):
    # ...

# Create the Predictor instance
predictor = Predictor(model, dataset_creator, dataloader_kwargs, save_predictions_fn=save_predictions_fn, output_transform=output_transform)

# Create and run the GUI application
app = predictor.create_app('Predictor App', 'org.example.predictor')
app.main_loop() 
```
## Running the Examples

To run the examples, you need to install the example dependencies separately:

1. Install the `speedygui` package: <br> ```pip install speedygui```
2. Navigate to the `examples` directory: <br> `cd /path/to/speedygui_project/examples`
3. Install the example dependencies: <br> ```pip install -r requirements.txt```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.
