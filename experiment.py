import csv
import os
from dataclasses import dataclass
from typing import Dict, List

from open_clip import CustomTextCLIP
from transformers import PreTrainedTokenizer

from coco_cn import get_coco_cn_data
from data_item import DataItem
from flickr30k import get_flickr30k_data
from flickr30k_200 import get_flickr30k_200_data
from flickr30k_cn import get_flickr30k_cn_data
from index import IndexType, build_index, get_recall
from xm3600 import get_xm3600_data
from xtd10 import get_xtd10_data
from xtd200 import get_xtd200_data


@dataclass
class LanguageRun:
    language: str
    index_type: str
    results: Dict[str, float]


@dataclass
class DatasetRun:
    dataset_name: str
    runs: List[LanguageRun]


class Experiment:
    def __init__(self, embedding_size: int, exp_name: str, save_path: str) -> None:
        self.embedding_size = embedding_size
        self.exp_name = exp_name
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.runs: List[DatasetRun] = []

    def run(self, model, transform, tokenizer):
        dataset_run = self.test_xtd10(model, transform, tokenizer)
        self.runs.append(dataset_run)
        dataset_run = self.test_coco_cn(model, transform, tokenizer)
        self.runs.append(dataset_run)
        dataset_run = self.test_xm3600(model, transform, tokenizer)
        self.runs.append(dataset_run)
        dataset_run = self.test_flickr30k_cn(model, transform, tokenizer)
        self.runs.append(dataset_run)
        dataset_run = self.test_flickr30k(model, transform, tokenizer)
        self.runs.append(dataset_run)
        dataset_run = self.test_xtd200(model, transform, tokenizer)
        self.runs.append(dataset_run)
        dataset_run = self.test_flickr30k_200(model, transform, tokenizer)
        self.runs.append(dataset_run)

    def save(self):
        metrics = ["recall@1", "recall@5", "recall@10"]
        file_path = os.path.join(self.save_path, f"{self.exp_name}.csv")
        with open(file_path, "w") as csvfile:
            writer = csv.writer(csvfile)
            header = [""]
            header.extend(metrics)
            writer.writerow(header)
            for dataset_run in self.runs:
                for lang_run in dataset_run.runs:
                    row = [
                        f"{dataset_run.dataset_name}-{lang_run.index_type}-{lang_run.language}"
                    ]
                    for metric in metrics:
                        row.append(lang_run.results[metric])
                    writer.writerow(row)

    def test_xtd10(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing XTD10 data")
        print("-----------")

        data_items = get_xtd10_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("xtd10", lang_runs)

    def test_xtd200(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing XTD200 data")
        print("-----------")

        data_items = get_xtd200_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("xtd200", lang_runs)

    def test_xm3600(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing Crossmodal3600 data")
        print("-----------")

        data_items = get_xm3600_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("crossmodal3600", lang_runs)

    def test_coco_cn(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing COCO CN data")
        print("-----------")

        data_items = get_coco_cn_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("coco_cn", lang_runs)

    def test_flickr30k_cn(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing Flickr30k CN data")
        print("-----------")

        data_items = get_flickr30k_cn_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("flickr30k_cn", lang_runs)

    def test_flickr30k(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing Flickr30k data, test split")
        print("-----------")

        data_items = get_flickr30k_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("flickr30k", lang_runs)

    def test_flickr30k_200(
        self,
        model,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> DatasetRun:
        print("-----------")
        print("Testing Flickr30k-200 data")
        print("-----------")

        data_items = get_flickr30k_200_data()
        print(f"Dataset size: {len(data_items)} items")
        lang_runs = self.run_test(data_items, model, transform, tokenizer)
        return DatasetRun("flickr30k-200", lang_runs)

    def run_test(
        self,
        data_items: List[DataItem],
        model: CustomTextCLIP,
        transform,
        tokenizer: PreTrainedTokenizer,
    ) -> List[LanguageRun]:
        langs = unique_languages(data_items)
        res = []
        index_types = [IndexType.TextToImage, IndexType.ImageToText]
        for index_type in index_types:
            index_type_str = "t2i" if index_type == IndexType.TextToImage else "i2t"
            for lang_code in langs:
                run = LanguageRun(
                    lang_code,
                    index_type_str,
                    {},
                )
                subset = []
                for data_item in data_items:
                    if data_item.language == lang_code:
                        subset.append(data_item)
                print(f"Subset size for {lang_code}: {len(subset)} items")
                index = build_index(
                    subset, model, transform, tokenizer, index_type, self.embedding_size
                )
                options = [1, 5, 10]
                recalls = get_recall(
                    subset, model, transform, tokenizer, index, options, index_type
                )
                for i, recall in enumerate(recalls):
                    print(
                        f"Type: {index_type_str}, language: {lang_code}, recall@{options[i]}: {recall}"
                    )
                    run.results[f"recall@{options[i]}"] = recall
                res.append(run)
        return res


def unique_languages(items: List[DataItem]):
    langs = set()
    for item in items:
        langs.add(item.language)
    return list(langs)
