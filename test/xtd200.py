import os
from typing import List

from data_item import DataItem
from flores_langs import flores_lang_codes
from xtd10 import get_file_paths, image_dir, image_names_file

data_dir = "../data/xtd200"


def get_xtd200_data() -> List[DataItem]:
    file_path = os.path.join(data_dir, image_names_file)
    with open(file_path, "r") as f:
        image_names = f.readlines()
    image_names = [name.replace("\n", "") for name in image_names]
    image_paths = get_file_paths(image_names, image_dir)
    res_items: List[DataItem] = []
    for lang in flores_lang_codes:
        filename = f"{lang}.txt"
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r") as f:
            captions = f.readlines()
        captions = [caption.replace("\n", "") for caption in captions]
        for i, _ in enumerate(image_paths):
            data_item = DataItem(
                image_path=image_paths[i],
                language=lang,
                captions=[captions[i]],
                eng_caption="",
                image=None,
            )
            res_items.append(data_item)
    return res_items
