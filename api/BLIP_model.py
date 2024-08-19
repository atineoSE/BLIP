from typing import Any

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


class BLIPModel:
    def __init__(self, model: Any, image_size: int, device: str | None):
        self.image_size = image_size
        self.device = device
        self.model = model
        self.model.eval()  # Set to evaluation mode
        if device:
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
