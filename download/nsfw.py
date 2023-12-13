from typing import List

import numpy as np
import onnxruntime
from PIL.Image import Image
from transformers import AutoProcessor


class NSFWPredictor:
    def __init__(self, model_path: str) -> None:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.ort_session = onnxruntime.InferenceSession(
            model_path, session_options, providers=["CPUExecutionProvider"]
        )
        self.processor = AutoProcessor.from_pretrained("Falconsai/nsfw_image_detection")

    def process(self, images: List[Image]) -> bool:
        inputs = self.processor(images=images, return_tensors="np")
        ort_inputs = {"pixel_values": inputs["pixel_values"]}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        prediction = np.argmax(ort_outputs[0], -1, keepdims=True)
        return prediction == 1
