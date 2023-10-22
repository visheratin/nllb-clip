import argparse
import multiprocessing as mp
import os
from time import sleep

from data_item import DataItem
from datasets import load_dataset
from downloader import init_download
from translator import init_translation

if __name__ == "__main__":
    endpoint_url = os.environ["S3_ENDPOINT_URL"]
    key_id = os.environ["S3_KEY_ID"]
    secret_key = os.environ["S3_SECRET_KEY"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        dest="batch_size",
        default=32,
    )
    parser.add_argument("-o", "--out_dir", type=str, dest="out_dir", default="./data")
    parser.add_argument("-b", "--bucket", type=str, dest="bucket", default="")
    parser.add_argument(
        "-i",
        "--img_dir",
        type=str,
        dest="img_dir",
        default="./images",
    )
    parser.add_argument("-df", "--data_file", type=str, dest="data_file", default="")
    parser.add_argument(
        "-tn", "--translators_num", type=int, dest="translators_num", default=1
    )
    parser.add_argument(
        "-dn", "--downloaders_num", type=int, dest="downloaders_num", default=1
    )
    parser.add_argument("-gn", "--gpu_num", type=int, dest="gpu_num", default=1)
    args = parser.parse_args()
    data_queue = mp.Queue()
    items_queue = mp.Queue()
    for i in range(args.downloaders_num):
        p = mp.Process(
            target=init_download,
            args=(
                data_queue,
                "./aesthetic-scorer.onnx",
                args.img_dir,
                items_queue,
                endpoint_url,
                key_id,
                secret_key,
                args.bucket,
            ),
        )
        p.start()
    for i in range(args.translators_num):
        p = mp.Process(
            target=init_translation,
            args=(
                i,
                "./nllb-200-3.3B-int8",
                "./flores200_sacrebleu_tokenizer_spm.model",
                items_queue,
                args.batch_size,
                args.out_dir,
                [i % args.gpu_num],
            ),
        )
        p.start()
    if args.data_file == "":
        dataset = load_dataset(
            "laion/laion-coco",
            split="train",
        )
    else:
        dataset = load_dataset(
            "laion/laion-coco",
            split="train",
            data_files=args.data_file,
        )
    for _, item in enumerate(dataset):
        data_item = DataItem("", item["URL"], item["top_caption"], 0)
        data_queue.put(data_item)

    print("Loaded all items into the queue")
    while True:
        sleep(1000)
