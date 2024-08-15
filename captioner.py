import os
from typing import Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from models.blip import blip_decoder


class Captioner:
    def __init__(self, image_size: int, device: str = "cuda"):
        self.image_size = image_size
        self.device = device

        model_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"
        self.model = blip_decoder(
            pretrained=model_url, image_size=image_size, vit="base"
        )
        self.model.eval()  # Set to evaluation mode
        self.model = self.model.to(device)

    def _load_image(self, image: Image) -> Any:
        transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
        image_tensor = transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        return image_tensor

    def get_caption(self, image: Image):
        image_tensor = self._load_image(image)

        with torch.no_grad():
            # beam search
            caption = self.model.generate(
                image_tensor, sample=False, num_beams=3, max_length=20, min_length=5
            )
            # nucleus sampling
            # caption = self.model.generate(image_tensor, sample=True, top_p=0.9, max_length=20, min_length=5)
            return caption
