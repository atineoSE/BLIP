import torch
from PIL import Image

from api.BLIP_model import BLIPModel
from models.blip_vqa import blip_vqa

VQA_MODEL_URL = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_vqa_capfilt_large.pth"


class Questioner(BLIPModel):
    def __init__(self, image_size: int, device: str | None = "cuda"):
        model = blip_vqa(pretrained=VQA_MODEL_URL, image_size=image_size, vit="base")
        super().__init__(model, image_size, device)

    def get_answer(self, image: Image, question: str):
        image_tensor = self._load_image(image)

        with torch.no_grad():
            answer = self.model(
                image_tensor, question, train=False, inference="generate"
            )
            return answer
