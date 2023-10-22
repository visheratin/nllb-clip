from enum import Enum
from typing import List

import faiss
import torch
from data_item import DataItem
from lang_map import lang_map
from open_clip import CustomTextCLIP
from PIL import Image
from transformers import PreTrainedTokenizer


class IndexType(Enum):
    TextToImage = 1
    ImageToText = 2


def build_index(
    items: List[DataItem],
    model: CustomTextCLIP,
    transform,
    tokenizer: PreTrainedTokenizer,
    index_type: IndexType,
    embedding_size: int,
):
    index = faiss.IndexFlatL2(embedding_size)
    batch_size = 512
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        if index_type == IndexType.TextToImage:
            vectors = process_image_batch(batch, model, transform)
        else:
            vectors = process_text_batch(batch, model, tokenizer)
        faiss.normalize_L2(vectors)
        index.add(vectors)
    return index


def get_recall(
    items: List[DataItem],
    model: CustomTextCLIP,
    transform,
    tokenizer: PreTrainedTokenizer,
    index: faiss.Index,
    ks: List[int],
    index_type: IndexType,
):
    batch_size = 512
    hits = []
    for i, _ in enumerate(ks):
        hits.append(0)
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        if index_type == IndexType.TextToImage:
            vectors = process_text_batch(batch, model, tokenizer)
        else:
            vectors = process_image_batch(batch, model, transform)
        faiss.normalize_L2(vectors)
        for u, k in enumerate(ks):
            _, neighbors = index.search(vectors, k)
            for j in range(neighbors.shape[0]):
                global_idx = i + j
                if global_idx in neighbors[j]:
                    hits[u] += 1
    recalls = [hit / len(items) for hit in hits]
    return recalls


def process_image_batch(items: List[DataItem], model, transform):
    images = []
    for item in items:
        if item.image is not None:
            images.append(transform(item.image))
        else:
            image = Image.open(item.image_path).convert("RGB")
            images.append(transform(image))
    input_tensors = torch.stack(images, dim=0).to("cuda")
    # input_tensors = transform(images).to("cuda")
    with torch.no_grad():
        output = model.encode_image(input_tensors).cpu().detach().numpy()
    return output


def process_text_batch(items: List[DataItem], model, tokenizer: PreTrainedTokenizer):
    input_ids = []
    for item in items:
        tokenizer.set_src_lang_special_tokens(lang_map[item.language])
        inputs = tokenizer.batch_encode_plus(
            [item.captions[0]],
            max_length=100,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids.append(inputs["input_ids"])
    input_ids = torch.cat(input_ids, 0).to("cuda")
    with torch.no_grad():
        output = model.encode_text(input_ids).cpu().detach().numpy()
    return output
