import torch
from PIL import Image

from api.BLIP_model import BLIPModel
from models.blip import blip_decoder

CAPTION_MODEL_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth"


class Captioner(BLIPModel):
    def __init__(self, image_size: int, device: str | None = "cuda"):
        model = blip_decoder(
            pretrained=CAPTION_MODEL_URL, image_size=image_size, vit="base"
        )
        super().__init__(model, image_size, device)

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
