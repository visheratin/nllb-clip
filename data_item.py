from dataclasses import dataclass
from typing import List, Union

from PIL import Image


@dataclass
class DataItem:
    image_path: str
    image: Union[Image.Image, None]
    language: str
    captions: List[str]
    eng_caption: str
