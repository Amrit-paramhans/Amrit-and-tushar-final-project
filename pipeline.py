import torch
from PIL import Image
from typing import Dict, Any

from blip import BLIP2Wrapper
from detr import DETRWrapper
from trocr import TrOCRWrapper
from llava import LLaVAWrapper
from vectorbc import VectorBCWrapper

class VisionLanguagePipeline:
    def __init__(self, device: str = None):
        """Initializes all model wrappers for unified processing."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.blip = BLIP2Wrapper(device=self.device)
        self.detr = DETRWrapper(device=self.device)
        self.trocr = TrOCRWrapper(device=self.device)
        self.llava = LLaVAWrapper(device=self.device)
        self.gpt2 = GPT2Reasoner(device=self.device)
        self.vectorbc = VectorBCWrapper(device=self.device)

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process an image through the full pipeline and return results.
        :param image: PIL Image
        :return: Dictionary with outputs from each module
        """
        results = {}

        # Step 1: BLIP2 - Generate caption
        caption = self.blip.generate_caption(image)
        results["caption"] = caption

        # Step 2: DETR - Object detection
        objects = self.detr.detect_objects(image)
        results["objects"] = objects

        # Step 3: TrOCR - OCR detection (e.g., signs, dashboard readings)
        ocr_text = self.trocr.read_text(image)
        results["ocr"] = ocr_text

        # Step 4: LLaVA - Multimodal reasoning
        reasoning_result = self.llava.reason(image, caption)
        results["llava_reasoning"] = reasoning_result

        # Step 6: VectorBC - Predict driving decisions
        driving_decision = self.vectorbc.predict_action(image, objects, caption)
        results["driving_action"] = driving_decision

        return results
