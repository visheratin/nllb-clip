from typing import List

from datasets import load_dataset
from tqdm import tqdm

from data_item import DataItem

# https://huggingface.co/datasets/nlphuji/flickr30k


def get_flickr30k_data() -> List[DataItem]:
    dataset = load_dataset("nlphuji/flickr30k")
    data_items = []
    for item in tqdm(dataset["test"]):
        if item["split"] != "test":
            continue
        data_item = DataItem(
            image=item["image"],
            image_path="",
            language="en",
            captions=item["caption"],
            eng_caption="",
        )
        data_items.append(data_item)
    return data_items
