import multiprocessing as mp
import uuid
import warnings
from io import BytesIO
from time import sleep

import boto3
import requests
from aesthetic import AestheticPredictor
from data_item import DataItem
from nsfw import NSFWPredictor
from PIL import Image


def init_download(
    in_queue: mp.Queue,
    predictor_model_path: str,
    nsfw_model_path: str,
    image_dir: str,
    out_queue: mp.Queue,
    s3_endpoint: str = "",
    s3_key_id: str = "",
    s3_secret: str = "",
    s3_bucket: str = "",
):
    try:
        skip_upload = False
        if s3_endpoint == "" or s3_key_id == "" or s3_secret == "" or s3_bucket == "":
            warnings.warn(
                "S3 credentials are not provided, the images will be only stored locally"
            )
            skip_upload = True
        s3_client = None
        if not skip_upload:
            s3_client = boto3.client(
                "s3",
                endpoint_url=s3_endpoint,
                aws_access_key_id=s3_key_id,
                aws_secret_access_key=s3_secret,
            )
        predictor = AestheticPredictor(predictor_model_path)
        nsfw_predictor = NSFWPredictor(nsfw_model_path)
        print("Downloader was started")
        while True:
            item = in_queue.get()
            if item is None:
                break
            ok = process_item(
                item, predictor, nsfw_predictor, image_dir, s3_bucket, s3_client
            )
            if ok:
                out_queue.put(item)
            while out_queue.qsize() > 1000:
                sleep(1)
    except Exception as e:
        print(e)


def process_item(
    item: DataItem,
    predictor: AestheticPredictor,
    nsfw_predictor: NSFWPredictor,
    image_dir: str,
    s3_bucket: str,
    s3_client,
) -> bool:
    try:
        size_threshold = 400
        response = requests.get(item.url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        if img.width < size_threshold or img.height < size_threshold:
            return False
        score = float(predictor.process([img]))
        if score < 5.0:
            return False
        item.score = score
        if nsfw_predictor.process([img]):
            return False
        id = str(uuid.uuid4())
        item.id = id
        file_path = f"{image_dir}/{id}.jpg"
        s3_path = f"{id}.jpg"
        img.save(file_path)
        if s3_client is not None:
            s3_client.upload_file(file_path, s3_bucket, s3_path)
        return True
    except:
        return False
