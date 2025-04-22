from PIL import Image
from typing import Dict, Any

from blip import BLIP2Wrapper
from detr import DETRWrapper
from trocr import TrOCRWrapper
from llava import LLaVAWrapper

from vectorbc import VectorBCWrapper
import torch


class LLM4ADPipeline:
    """
    This is the main class responsible for orchestrating all components of the vision-language
    pipeline for interpretable autonomous driving decisions. The pipeline processes an image
    (captured from a dashcam, webcam, or uploaded file) and returns various driving-related outputs
    including captions, object detection results, OCR text, reasoning, and predicted driving actions.
    """

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes all the model wrappers for the complete pipeline.
        It will automatically use a CUDA-enabled device if available, otherwise, it defaults to CPU.

        :param device: The device to run the models (either "cuda" or "cpu").
        """
        self.device = device

        # Initializing each model wrapper
        self.blip = BLIP2Wrapper(device=self.device)  # Captioning model
        self.detr = DETRWrapper(device=self.device)  # Object detection model
        self.trocr = TrOCRWrapper(device=self.device)  # OCR model for detecting text
        self.llava = LLaVAWrapper(device=self.device)  # Multimodal reasoning model
        self.gpt2 = GPT2Reasoner(device=self.device)  # Text-based reasoning model
        self.vectorbc = VectorBCWrapper(device=self.device)  # Behavior cloning model for driving decisions

    def explain_driving_scene(self, image: Image.Image) -> Dict[str, Any]:
        """
        This method processes an image through the entire pipeline and provides
        a detailed human-understandable reasoning for driving decisions.

        The function does the following:
        1. Generates a caption from the image.
        2. Detects objects within the image.
        3. Reads any text from the image (OCR).
        4. Applies LLaVA for multimodal reasoning.
        5. Uses GPT-2 for generating reasoning text.
        6. Predicts driving decisions using VectorBC.

        :param image: A PIL Image object, typically representing a frame from a video or an uploaded image.
        :return: A dictionary containing the processed results, including captions, object detections, and reasoning.
        """
        results = {}

        # Step 1: Generate a caption using BLIP2
        caption = self.blip.generate_caption(image)
        results["caption"] = caption  # Store the caption for later use

        # Step 2: Detect objects in the image using DETR
        objects = self.detr.detect_objects(image)
        results["objects"] = objects  # List of detected objects with class labels

        # Step 3: Read text from the image (e.g., signs, labels, or dashboard) using TrOCR
        ocr_text = self.trocr.read_text(image)
        results["ocr"] = ocr_text  # Store any text detected in the image (e.g., road signs)

        # Step 4: Apply LLaVA model for multimodal reasoning, combining caption and image
        multimodal_reasoning = self.llava.reason(image, caption)
        results["llava_reasoning"] = multimodal_reasoning  # Logical reasoning based on image and caption

        # Step 5: Use GPT-2 for generating reasoning based on detected objects and OCR
        reasoning_text = self.gpt2.generate_reasoning(caption, objects, ocr_text)
        results["gpt2_reasoning"] = reasoning_text  # Reasoning based on visual and textual input

        # Step 6: Predict driving actions using the VectorBC model
        driving_action = self.vectorbc.predict_action(image, objects, caption)
        results["driving_action"] = driving_action  # Suggested driving action based on analysis

        # Step 7: Final human-readable structured report
        # Combining all the results in a structured format
        results["final_report"] = (
            f"🚘 **Driving Scene Caption**: {caption}\n\n"
            f"📦 **Detected Objects**: {[obj['label'] for obj in objects]}\n\n"
            f"🔤 **Detected Signs (OCR)**: {ocr_text}\n\n"
            f"🧠 **LLaVA Multimodal Reasoning**: {multimodal_reasoning}\n\n"
            f"➡️ **Suggested Driving Action**: {driving_action}\n\n"
            f"🔧 **Further Insights**: Based on the image, objects, and OCR, this action seems appropriate. "
            f"Consider whether road conditions or other vehicles are influencing this prediction."
        )

        # Optional: You can also log or store intermediate results for further analysis.
        # For example, store the detailed results for each model to analyze later
        results["log"] = {
            "blip_caption": caption,
            "detr_objects": objects,
            "trocr_ocr": ocr_text,
            "llava_reasoning": multimodal_reasoning,
            "vectorbc_action": driving_action
        }

        # Return the full results with all processed information
        return results

    def run_pipeline_for_video(self, video_frames: list) -> list:
        """
        This function processes a list of video frames and returns reasoning and predicted actions for each frame.

        :param video_frames: A list of PIL Image frames (e.g., from a dashcam or video stream)
        :return: A list of dictionaries, each containing processed results for a single frame
        """
        video_results = []

        # Process each frame in the video stream
        for frame in video_frames:
            frame_results = self.explain_driving_scene(frame)
            video_results.append(frame_results)  # Append results for each frame

        return video_results

    def process_and_generate_action_for_single_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        A helper function to process a single image and return only the predicted driving action.

        This can be useful when the user just needs to know the action based on the current image.

        :param image: PIL Image
        :return: Dictionary containing only the predicted driving action
        """
        results = self.explain_driving_scene(image)
        return {"driving_action": results["driving_action"]}

    def summarize_results(self, results: Dict[str, Any]) -> str:
        """
        Create a clean and concise human-readable summary of the results for display in a UI.

        :param results: The full results dictionary generated by explain_driving_scene()
        :return: A string summarizing the findings
        """
        return (
            f"**Driving Scene Summary**:\n\n"
            f"**Caption**: {results.get('caption', 'No caption available')}\n"
            f"**Detected Objects**: {results.get('objects', 'No objects detected')}\n"
            f"**OCR Text**: {results.get('ocr', 'No OCR text found')}\n"
            f"**LLaVA Reasoning**: {results.get('llava_reasoning', 'No reasoning available')}\n"
         
            f"**Predicted Action**: {results.get('driving_action', 'No action predicted')}\n"
        )

