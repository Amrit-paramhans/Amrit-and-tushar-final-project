import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from typing import Optional, Dict, Any


class BLIP2Wrapper:
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", device: Optional[str] = None):
        """
        Initialize the BLIP2 model and processor.
        :param model_name: HuggingFace model ID for BLIP2.
        :param device: 'cuda' or 'cpu'. Auto-detected if not provided.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(self.device)

    def generate_caption(self, image: Image.Image, prompt: Optional[str] = None, return_logits: bool = False) -> str:
        """
        Generate a caption for an input image.
        :param image: PIL Image to process.
        :param prompt: Optional natural language prompt (e.g., "Describe the traffic situation").
        :param return_logits: If True, also return raw logits for further reasoning modules.
        :return: Caption string (and optionally logits if requested).
        """
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if return_logits:
            with torch.no_grad():
                outputs = self.model(**inputs)
            return caption, outputs.logits

        return caption

    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        Extract hidden features (embeddings) from the image using BLIP2 encoder.
        This can be used as input to downstream tasks like LLaVA, GPT, or VectorBC.
        :param image: Input PIL image.
        :return: Dictionary of encoder features.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_embeds = self.model.vision_model(**inputs).last_hidden_state
        return {"vision_features": vision_embeds}
