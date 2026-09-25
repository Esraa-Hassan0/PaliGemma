from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np
import torch
from PIL import Image

IMAGENET_STD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD_STD = [0.5, 0.5, 0.5]


def add_img_tokens_to_prompt(prefix_prompt, bos_token, img_seq_len, img_token):
    """
    Prepends the required number of image tokens and the BOS token to the text prompt
    """
    return f"{img_token * img_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(img: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Rescales the pixel values of the image array by a given scale factor
    """
    rescaled_img = img * scale
    rescaled_img = rescaled_img.astype(dtype)
    return rescaled_img


def resize(
    img: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> Image.Image:
    """
    Resizes the PIL image to the specified (height, width)
    """
    h, w = size
    resized_image = img.resize((w, h), resample=resample, reducing_gap=reducing_gap)
    return resized_image


def normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    """
    Normalizes the image tensor using the provided mean and standard deviation
    """
    mean = np.array(mean, dtype=img.dtype)
    std = np.array(std, dtype=img.dtype)
    img = (img - mean) / std
    return img


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    rescale_factor: float = None,
    resample: Image.Resampling = None,
    image_mean: Optional[Union[List[float], float]] = None,
    image_std: Optional[Union[List[float], float]] = None,
) -> List[np.ndarray]:
    """
    Applies a series of preprocessing steps to a list of PIL images:
    resize, convert to numpy, rescale, normalize, and transpose to CHW format
    """
    h, w = size[0], size[1]
    images = [image.convert("RGB") if image.mode != "RGB" else image for image in images]
    images = [resize(img=image, size=(h, w), resample=resample) for image in images]

    images = [np.array(image) for image in images]

    images = [rescale(image, scale=rescale_factor) for image in images]

    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    images = [image.transpose(2, 0, 1) for image in images]

    return images


class PaliGemmaProcessor:

    """
    Processor for PaliGemma, handling both text tokenization and image preprocessing
    Combines pixel values and text input IDs into a unified dictionary for model ingestion
    """
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_img_tokens: int, img_size: int):
        super().__init__()
        self.image_size = img_size
        self.image_seq_length = num_img_tokens

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # For object detection (bounding boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]  # For object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        if hasattr(tokenizer, "add_bos_token"):
            tokenizer.add_bos_token = False  # we will add them ourselves later
        if hasattr(tokenizer, "add_eos_token"):
            tokenizer.add_eos_token = False  # we will add them ourselves later

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = False,
    ) -> dict:
        """
        Processes a pair of (texts, images) to return padded input tensors.
        """
        assert (
            len(images) == 1 and len(text) == 1
        ), f"Recieved {len(images)} for {len(text)} prompts."

        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
            image_mean=IMAGENET_STD_MEAN,
            image_std=IMAGENET_STD_STD,
        )

        # Convert the list of numpy arrays into a single numpy array with shape [B, C, H, W]
        pixel_values = np.stack(pixel_values, axis=0)

        # Convert the numpy array to a PyTorch tensor
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

        # Prepend a `self.image_seq_len` number of image tokens to the prompt
        input_strings = [
            add_img_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                img_seq_len=self.image_seq_length,
                img_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]
        # return the input_ids and attention_mask as PyTorch tensors
        inputs = self.tokenizer(
            input_strings, return_tensors="pt", padding=padding, truncation=truncation
        )
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data
