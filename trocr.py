from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
from typing import List


class TrOCRWrapper:
    """
    This class wraps around the TrOCR (Text Recognition) model from Hugging Face.
    It is responsible for recognizing text from images, such as road signs, license plates,
    and other text information that may be visible in a driving scene.

    Attributes:
        processor: The TrOCR processor used to preprocess the input image for the model.
        model: The pre-trained TrOCR model for text generation.
        device: The device (CPU or CUDA) to run the model on.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the TrOCR model and processor. The processor tokenizes the input image,
        and the model performs the OCR (Optical Character Recognition).

        :param device: The device to run the models (either "cuda" or "cpu").
        """
        self.device = device
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-structured")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-structured")

        # Move the model to the appropriate device (CPU or GPU)
        self.model.to(self.device)

    def read_text(self, image: Image.Image) -> List[str]:
        """
        Processes the given image and extracts any text using the TrOCR model.
        The function returns a list of strings containing the detected text in the image.

        :param image: A PIL Image object that may contain text.
        :return: A list of strings representing detected text.
        """
        # Preprocess the image for the TrOCR model
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Perform inference (OCR)
        with torch.no_grad():
            generated_ids = self.model.generate(input_ids=inputs.pixel_values)

        # Decode the generated tokens into text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_text

    def visualize_ocr_results(self, image: Image.Image, detected_text: List[str]) -> Image:
        """
        Visualizes the OCR results by overlaying the detected text on the input image.
        This can be helpful for debugging and understanding what the OCR model is detecting.

        :param image: The original image containing the detected text.
        :param detected_text: A list of strings representing the detected text in the image.
        :return: The image with text overlaid.
        """
        from PIL import ImageDraw, ImageFont

        # Create a drawing context to add text to the image
        draw = ImageDraw.Draw(image)

        # Use a basic font (you can change this to a custom one if desired)
        font = ImageFont.load_default()

        # Overlay the detected text at the top of the image
        y_position = 10  # Starting Y position for text
        for text in detected_text:
            draw.text((10, y_position), text, fill="white", font=font)
            y_position += 20  # Move the next line of text down

        return image

    def extract_sign_text(self, image: Image.Image) -> str:
        """
        A specific function for extracting and returning road sign text from an image.
        This function is customized for scenarios where road signs are the primary text of interest.

        :param image: A PIL Image object representing the road sign or image containing signs.
        :return: A string of the detected text on the road sign.
        """
        detected_text = self.read_text(image)
        # In this case, we can return the first detected text (this could be a road sign)
        return detected_text[0] if detected_text else "No text detected"

