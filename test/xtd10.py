import os
from typing import List

from data_item import DataItem
from tqdm import tqdm

image_names_file = "test_image_names.txt"
data_dir = "../data/xtd10/captions"
image_dir = "../data/xtd10/images"


def get_xtd10_data() -> List[DataItem]:
    file_path = os.path.join(data_dir, image_names_file)
    with open(file_path, "r") as f:
        image_names = f.readlines()
    image_names = [name.replace("\n", "") for name in image_names]
    image_paths = get_file_paths(image_names, image_dir)
    res_items: List[DataItem] = []
    for lang in lang_codes:
        filename = f"test_1kcaptions_{lang}.txt"
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


def get_file_paths(item_names: List[str], image_dir: str):
    train_image_names = os.listdir(image_dir)
    train_image_names = [name for name in train_image_names if name in item_names]
    train_image_paths = [os.path.join(image_dir, name) for name in train_image_names]
    image_paths = train_image_paths
    image_filenames = [os.path.basename(path) for path in image_paths]
    res = []
    for item in tqdm(item_names):
        try:
            idx = image_filenames.index(item)
            res.append(image_paths[idx])
        except:
            continue
    return res


lang_codes = ["en", "es", "it", "ko", "pl", "ru", "tr", "zh", "de", "fr", "jp"]
