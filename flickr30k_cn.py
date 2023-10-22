import os
from typing import List

from datasets import load_dataset
from tqdm import tqdm

from data_item import DataItem

# https://zero.so.com/download.html

test_file_path = "./flickr30k-cn/test.txt"
val_file_path = "./flickr30k-cn/val.txt"


def get_flickr30k_cn_data() -> List[DataItem]:
    dataset = load_dataset("nlphuji/flickr30k")
    test_items = read_file(test_file_path)
    val_items = read_file(val_file_path)
    data_items = test_items + val_items
    image_names = [item.image_path for item in data_items]
    filtered_dataset = dataset.filter(
        lambda example: example["filename"] in image_names
    )
    for item in tqdm(filtered_dataset["test"]):
        for data_item in data_items:
            if item["filename"] == data_item.image_path:
                data_item.image = item["image"]
                data_item.image_path = ""
                data_item.eng_caption = item["caption"][0]
                break
    data_items = [item for item in data_items if item.image is not None]
    return data_items


def read_file(file_path: str) -> List[DataItem]:
    items: List[DataItem] = []
    with open(file_path, "r") as f:
        for line in f:
            if len(line) == 0:
                continue
            line_parts = line.strip().split("\t")
            info_parts = line_parts[0].split("#")
            if len(info_parts) == 1:
                image_id = line_parts[0]
                language = "zhm"
            else:
                image_id = info_parts[0]
                language = info_parts[1]
            image_name = image_id + ".jpg"
            caption = line_parts[1]
            found = False
            for item in items:
                if item.image_path == image_name:
                    item.captions.append(caption)
                    found = True
                    break
            if not found:
                item = DataItem(
                    image_path=image_name,
                    language=language,
                    captions=[caption],
                    eng_caption="",
                    image=None,
                )
                items.append(item)
    return items
