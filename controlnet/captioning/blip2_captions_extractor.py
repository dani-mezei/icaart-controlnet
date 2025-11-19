from typing import List

import torch
import numpy as np
from PIL import Image

from transformers import Blip2Processor, Blip2ForConditionalGeneration


class Blip2CaptionsExtractor:
    def __init__(self, blip2_path: str, device: torch.device = None):
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self._device = device
        self._torch_dtype = torch.float32
        self._processor = Blip2Processor.from_pretrained(blip2_path)
        self._captioning_model = Blip2ForConditionalGeneration.from_pretrained(
            blip2_path,
            torch_dtype=self._torch_dtype,
        ).to(self._device)

    def preprocess_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image = np.asarray(image)
        return image

    def extract(
        self,
        image_paths: List[str],
        prompt: str = None,
        max_new_tokens: int = 50,
    ):
        if prompt is not None:
            prompt = [prompt for _ in range(len(image_paths))]

        images = [self.preprocess_image(image_path)
                  for image_path in image_paths]

        inputs = self._processor(
            images=images,
            text=prompt,
            return_tensors="pt",
        ).to(self._device, self._torch_dtype)

        generated_ids = self._captioning_model.generate(
            **inputs, max_new_tokens=max_new_tokens
        )
        generated_texts = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        generated_texts = [text.strip() for text in generated_texts]
        return generated_texts
