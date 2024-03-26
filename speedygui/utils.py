import torch
from PIL import Image
from skimage import color
from torchvision.transforms.v2.functional import to_pil_image


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverses the normalization on a tensor."""
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def tensor_to_image(tensor, normalize=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Convert a tensor to a PIL image, optionally applying denormalization.

    Args:
        tensor (torch.Tensor): The tensor to convert.
        normalize (bool): Whether to apply denormalization.
        mean (list): The mean used for normalization.
        std (list): The standard deviation used for normalization.

    Returns:
        PIL.Image: The converted PIL image.
    """
    if tensor.shape[0] != 3:
        raise ValueError("Input tensor must have 3 channels (RGB).")

    if normalize:
        tensor = denormalize(tensor, mean=mean, std=std)
    # Ensure tensor is in the range [0, 1] for conversion
    tensor = tensor.clamp(0, 1)
    return to_pil_image(tensor)


def mask_to_image(tensor, bg_label=0):
    """
        Convert a tensor representing a segmentation mask to a PIL image with label colors.

        Args:
            tensor (torch.Tensor): The tensor representing the segmentation mask.
            bg_label (int, optional): The label value to be treated as the background. Defaults to 0.

        Returns:
            PIL.Image: The converted PIL image with label colors.
    """

    if tensor.ndim == 3:  # C x H x W
        np_tensor = tensor.permute(1, 2, 0).cpu().numpy()
    else:  # H x W
        np_tensor = tensor.cpu().numpy()

    # Use label2rgb to convert label ids to colors
    color_pred = color.label2rgb(np_tensor, bg_label=bg_label)
    mask_image = Image.fromarray((color_pred * 255).astype("uint8"))
    return mask_image


