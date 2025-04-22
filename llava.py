from transformers import LLaVAProcessor, LLaVAModel
import torch
from typing import List


class LLaVAWrapper:
    """
    This class wraps around the LLaVA (Large Language and Vision Assistant) model.
    LLaVA is used for reasoning based on both visual (image) and textual (caption) inputs.

    Attributes:
        processor: The LLaVA processor used to preprocess inputs (images, captions).
        model: The pre-trained LLaVA model for vision and language reasoning.
        device: The device (CPU or GPU) to run the model on.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the LLaVA model and processor. The processor prepares the image and caption,
        and the model performs reasoning based on both.

        :param device: The device to run the models (either "cuda" or "cpu").
        """
        self.device = device
        # Load the LLaVA processor and model from Hugging Face
        self.processor = LLaVAProcessor.from_pretrained("llava/llava-v1")
        self.model = LLaVAModel.from_pretrained("llava/llava-v1")

        # Move the model to the appropriate device (CPU or GPU)
        self.model.to(self.device)

    def reason(self, image: torch.Tensor, caption: str) -> str:
        """
        Reason about the visual and textual inputs using the LLaVA model.
        The model performs reasoning based on the provided image and caption.

        :param image: A tensor representing the processed image (usually after being preprocessed).
        :param caption: A string representing the caption or textual description of the image.
        :return: The reasoned output based on the image and caption.
        """
        # Preprocess the inputs: image and caption
        inputs = self.processor(images=image, text=caption, return_tensors="pt").to(self.device)

        # Perform inference (reasoning) using the LLaVA model
        with torch.no_grad():
            outputs = self.model.generate(**inputs)

        # Decode and return the output
        reasoned_text = self.processor.decode(outputs[0], skip_special_tokens=True)
        return reasoned_text

    def visualize_reasoning(self, image: torch.Tensor, caption: str) -> str:
        """
        Visualize the reasoning output by simply returning the text along with the image caption.
        This method can be used to show the reasoning process or output along with the input caption.

        :param image: A tensor representing the processed image.
        :param caption: A string representing the caption or description for the image.
        :return: A string combining the caption and the reasoned output.
        """
        reasoned_output = self.reason(image, caption)
        return f"Caption: {caption}\nReasoning: {reasoned_output}"
