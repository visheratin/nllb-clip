import codecs
import csv
import multiprocessing as mp
import os
import time
from typing import List

import ctranslate2
import sentencepiece as spm
from data_item import DataItem
from langs import langs


def init_translation(
    id: int,
    model_path: str,
    tokenizer_path: str,
    queue: mp.Queue,
    batch_size: int,
    out_dir: str,
    device_index: List[int],
):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    model = ctranslate2.Translator(model_path, device="cuda", device_index=device_index)
    data_dir = os.path.join(out_dir, str(id))
    os.makedirs(data_dir, exist_ok=True)
    print(f"Translator {id} was started")
    c = 0
    while True:
        items = []
        for _ in range(batch_size):
            item = queue.get()
            if item is None:
                break
            items.append(item)
        if len(items) == 0:
            continue
        start = time.time()
        translate(c, items, model, tokenizer, data_dir)
        end = time.time()
        print(
            f"Translator {id} translated {len(items)} items in {int(end - start)} seconds"
        )
        c += 1


def translate(
    id: int,
    items: List[DataItem],
    model: ctranslate2.Translator,
    tokenizer: spm.SentencePieceProcessor,
    data_dir: str,
):
    captions = [item.caption for item in items]
    inputs = tokenizer.encode_as_pieces(captions)
    inputs = [["eng_Latn"] + sent + ["</s>"] for sent in inputs]
    file_path = os.path.join(data_dir, f"{id}.csv")
    fp = codecs.open(file_path, "w", "utf-8")
    writer = csv.writer(fp)
    for lang in langs:
        target_prefix = [[lang]] * len(captions)
        translations_subworded = model.translate_batch(
            inputs,
            batch_type="tokens",
            max_batch_size=2024,
            beam_size=4,
            target_prefix=target_prefix,
        )
        translations_subworded = [
            translation[0]["tokens"] for translation in translations_subworded
        ]
        for translation in translations_subworded:
            if lang in translation:
                translation.remove(lang)
        output = tokenizer.decode(translations_subworded)
        for i, item in enumerate(items):
            item = items[i]
            translated = output[i]
            row = [item.id, item.url, lang, translated, item.caption, item.score]
            writer.writerow(row)
    fp.close()
