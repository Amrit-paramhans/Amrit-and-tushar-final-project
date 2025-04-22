import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
from typing import List, Dict, Optional


class DETRWrapper:
    def __init__(self, model_name: str = "facebook/detr-resnet-50", device: Optional[str] = None):
        """
        Initialize DETR model for object detection.
        :param model_name: HuggingFace model ID.
        :param device: Device string: 'cuda' or 'cpu'.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name).to(self.device)

    def detect_objects(self, image: Image.Image, threshold: float = 0.7) -> List[Dict[str, any]]:
        """
        Detect objects in the image using DETR.
        :param image: PIL image to process.
        :param threshold: Confidence threshold to filter predictions.
        :return: List of detected objects with label, score, and bounding box.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (H, W)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": self.model.config.id2label[label.item()],
                "score": round(score.item(), 3),
                "box": [round(coord, 2) for coord in box.tolist()]  # [x0, y0, x1, y1]
            })

        return detections

    def get_raw_outputs(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """
        Return raw model outputs (logits, box predictions, etc.) for advanced usage.
        :param image: PIL image.
        :return: Dictionary of raw outputs.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
