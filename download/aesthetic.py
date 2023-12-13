from typing import List

import onnxruntime
from PIL.Image import Image
from transformers import AutoProcessor


class AestheticPredictor:
    def __init__(self, model_path: str) -> None:
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        self.ort_session = onnxruntime.InferenceSession(
            model_path, session_options, providers=["CPUExecutionProvider"]
        )
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def process(self, images: List[Image]):
        inputs = self.processor(images=images, return_tensors="np")
        ort_inputs = {"input_tensor": inputs["pixel_values"]}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        return ort_outputs[0]
