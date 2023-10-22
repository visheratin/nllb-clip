import json
import os
from typing import List

from data_item import DataItem
from tqdm import tqdm

# https://google.github.io/crossmodal-3600/

captions_path = "../data/xm3600/captions.jsonl"
image_dir = "../data/xm3600/images"


def get_xm3600_data() -> List[DataItem]:
    with open(captions_path, "r") as f:
        data = f.readlines()
    data = [json.loads(line) for line in data]
    res_items: List[DataItem] = []
    for data_item in tqdm(data):
        image_id = data_item["image/key"]
        image_name = f"{image_id}.jpg"
        for lang in lang_codes:
            captions = data_item[lang]["caption"]
            item = DataItem(
                image_path=os.path.join(image_dir, image_name),
                language=lang,
                captions=captions,
                eng_caption="",
                image=None,
            )
            res_items.append(item)
    return res_items


lang_codes = [
    "ar",
    "bn",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fa",
    "fi",
    "fil",
    "fr",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "he",
    "ja",
    "ko",
    "mi",
    "nl",
    "no",
    "pl",
    "pt",
    "quz",
    "ro",
    "ru",
    "sv",
    "sw",
    "te",
    "th",
    "tr",
    "uk",
    "vi",
    "zh",
]
