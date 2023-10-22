import multiprocessing as mp
import uuid
import warnings
from io import BytesIO
from time import sleep

import requests
from data_item import DataItem
from PIL import Image
from predictor import AestheticPredictor


def init_download(
    in_queue: mp.Queue,
    predictor_model_path: str,
    image_dir: str,
    out_queue: mp.Queue,
    s3_endpoint: str = "",
    s3_key_id: str = "",
    s3_secret: str = "",
    s3_bucket: str = "",
):
    skip_upload = False
    if s3_endpoint == "" or s3_key_id == "" or s3_secret == "" or s3_bucket == "":
        warnings.warn(
            "S3 credentials are not provided, the images will be only stored locally"
        )
        skip_upload = True
    s3_client = None
    if not skip_upload:
        import boto3

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_key_id,
            aws_secret_access_key=s3_secret,
        )
    predictor = AestheticPredictor(predictor_model_path)
    print("Downloader was started")
    while True:
        item = in_queue.get()
        if item is None:
            break
        ok = process_item(item, predictor, image_dir, s3_bucket, s3_client)
        if ok:
            out_queue.put(item)
        while out_queue.qsize() > 1000:
            sleep(1)


def process_item(
    item: DataItem,
    predictor: AestheticPredictor,
    image_dir: str,
    s3_bucket: str,
    s3_client,
) -> bool:
    try:
        response = requests.get(item.url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        score = float(predictor.process([img]))
        if score < 4.5:
            return False
        item.score = score
        id = str(uuid.uuid4())
        item.id = id
        file_path = f"{image_dir}/{id}.jpg"
        s3_path = f"{id}.jpg"
        img.save(file_path)
        if s3_client is not None:
            s3_client.upload_file(file_path, s3_bucket, s3_path)
        return True
    except Exception as e:
        print(e)
        return False
