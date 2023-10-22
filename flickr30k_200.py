import os
from typing import List

from data_item import DataItem
from flickr30k import get_flickr30k_data
from flores_langs import flores_lang_codes

data_dir = "./data/flickr30k-200"


def get_flickr30k_200_data() -> List[DataItem]:
    flickr30k_items = get_flickr30k_data()
    res_items: List[DataItem] = []
    for lang in flores_lang_codes:
        filename = f"{lang}.txt"
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r") as f:
            captions = f.readlines()
        captions = [caption.replace("\n", "") for caption in captions]
        for i, item in enumerate(flickr30k_items):
            data_item = DataItem(
                image_path="",
                language=lang,
                captions=[captions[i]],
                eng_caption="",
                image=item.image,
            )
            res_items.append(data_item)
    return res_items
