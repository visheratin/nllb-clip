import json
import os
from typing import List

from tqdm import tqdm

from data_item import DataItem

# https://github.com/li-xirong/coco-cn

test_file_path = "./data/coco_cn/captions.json"


def get_coco_cn_data() -> List[DataItem]:
    with open(test_file_path, "r") as f:
        data = json.load(f)
    images_info = data["images"]
    annotations = data["annotations"]
    image_ids = {}
    for item in images_info:
        image_ids[item["id"]] = item["file_name"]
    res_items = []
    for item in tqdm(annotations):
        filename = image_ids[item["image_id"]]
        language = item["source"]
        caption = item["caption"]
        found = False
        for res_item in res_items:
            if res_item.image_path == filename and res_item.language == language:
                res_item.captions.append(caption)
                found = True
                break
        if not found:
            data_item = DataItem(
                image_path=f"./data/coco_cn/images/{filename}",
                language=language,
                captions=[caption],
                eng_caption="",
                image=None,
            )
            res_items.append(data_item)
    return res_items
