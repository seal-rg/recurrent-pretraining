"""Data gen script for OLCF pretraing

this dataset is breaking me

"""

from pydantic import BaseModel
from typing import List, Optional, TYPE_CHECKING
import yaml
import os
from datetime import datetime

from pathlib import Path
import json
import boto3

from botocore import UNSIGNED
from botocore.config import Config

import logging
import traceback


from concurrent.futures import ThreadPoolExecutor, as_completed
from smart_open import open
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from datasets import Dataset
import multiprocessing as mp
from tqdm.auto import tqdm
from functools import partial
from botocore.exceptions import BotoCoreError, ClientError

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # cant make this work for now


# from transformers import PreTrainedTokenizerFast


# tokenizer = PreTrainedTokenizerFast("tomg-group-umd/huginn_tokenizer_test")
# tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|begin_header|>' + message['role'] + '<|end_header|>\n\n'+ message['content'] | trim + '<|end_turn|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% endif %}"

# current consts:
BASE_DIR = "/lustre/orion/csc569/scratch/jgeiping/data"
DOWNLOADED_DATASETS_PATH = f"{BASE_DIR}/test_download_folder"  # /test_download_folder
CACHE_DIR = f"{BASE_DIR}/test_cache"
FINAL_LOCATION = f"{BASE_DIR}/test_processed_dataset"
LIMITER = 2**18  # how many rows to take from each dataset in streaming mode
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = CACHE_DIR

import datasets
import huggingface_hub

# somehow required otherwise the token is not always read correctly:
huggingface_hub.login(token=os.environ.get("MANUALLY_SET_HF_TOKEN", None), add_to_git_credential=True)

if TYPE_CHECKING:
    import datasets.config


num_proc = mp.cpu_count()
datasets.config.STREAMING_READ_MAX_RETRIES = 50
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

S3_CONFIG = Config(
    signature_version=UNSIGNED,
    retries={"max_attempts": 16, "mode": "adaptive"},
    max_pool_connections=128,
)


class FileManager:
    def __init__(self, output_dir, file_size_limit):
        self.output_dir = output_dir
        self.file_size_limit = file_size_limit
        self.current_file = None
        self.current_file_size = 0
        self.current_file_index = 0

    def write(self, data):
        if self.current_file is None or self.current_file_size + len(data) > self.file_size_limit:
            self._open_new_file()
        self.current_file.write(data)  # type: ignore
        self.current_file_size += len(data)

    def _open_new_file(self):
        if self.current_file:
            self.current_file.close()
        self.current_file_index += 1
        filename = os.path.join(self.output_dir, f"processed_items_{self.current_file_index}.jsonl")
        self.current_file = open(filename, "w")
        self.current_file_size = 0

    def close(self):
        if self.current_file:
            self.current_file.close()


@retry(
    stop=stop_after_attempt(10),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((BotoCoreError, ClientError, IOError)),
)
def download_file(s3, repo_file):
    s3_url = f"s3://softwareheritage/content/{repo_file['blob_id']}"
    try:
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            return fin.read().decode(repo_file["src_encoding"])
    except UnicodeDecodeError as e:
        logger.warning(
            f"Unicode decode error for {s3_url}: {str(e)}. Attempting to decode as utf-8 with errors ignored."
        )
        with open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            return fin.read().decode("utf-8", errors="ignore")


def process_files(s3, files, max_workers=32):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(download_file, s3, repo_file): repo_file for repo_file in files}
        results = []
        for future in as_completed(future_to_file):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Failed to process file: {str(e)}")
        return "\n".join(results)


@retry(stop=stop_after_attempt(3), wait=wait_random_exponential(multiplier=1, max=60))
def process_batch(batch, max_workers):
    try:
        s3 = boto3.client("s3", config=S3_CONFIG)
        results = []
        for item in batch:
            try:
                content = process_files(s3, item["files"], max_workers)
                results.append({"id": item["id"], "content": content})
            except Exception as e:
                logger.error(f"Failed to process item {item['id']}: {str(e)}")
        return results
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}\n{traceback.format_exc()}")
        raise  # Re-raise the exception to trigger a retry


def software_heritage_aws_download(
    dataset, batch_size=1024, output_dir="output", file_size_limit=1024 * 1024 * 1024, num_proc=mp.cpu_count()
):
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, "progress.json")

    try:
        with open(progress_file, "r") as f:
            progress_data = json.load(f)
        processed_batches = set(progress_data["processed_batches"])
        current_file_index = progress_data["current_file_index"]
    except (FileNotFoundError, json.JSONDecodeError):
        processed_batches = set()
        current_file_index = 0

    file_manager = FileManager(output_dir, file_size_limit)
    file_manager.current_file_index = current_file_index

    total_batches = (len(dataset) + batch_size - 1) // batch_size
    batches = [dataset.select(range(i, min(i + batch_size, len(dataset)))) for i in range(0, len(dataset), batch_size)]

    process_func = partial(process_batch, max_workers=32)

    with mp.Pool(num_proc) as pool:
        batch_iterator = pool.imap(process_func, batches)
        for batch_index, batch_results in tqdm(
            enumerate(batch_iterator), total=total_batches, desc="Processing batches"
        ):
            if batch_index * batch_size in processed_batches:
                continue  # Skip already processed batches

            for result in batch_results:
                file_manager.write(json.dumps(result) + "\n")

            processed_batches.add(batch_index * batch_size)

            # Update progress file
            with open(progress_file, "w") as progress_f:
                json.dump(
                    {
                        "processed_batches": list(processed_batches),
                        "current_file_index": file_manager.current_file_index,
                    },
                    progress_f,
                )

    file_manager.close()

    return Dataset.from_json(os.path.join(output_dir, "processed_items_*.jsonl"))


class DataSource(BaseModel):
    address: str
    subset: Optional[str] = None
    features: List[str]
    needs_chat_templating: bool = False
    license: Optional[str] = None
    citation: Optional[str] = None
    machine_generated: bool = False
    requires_software_heritage_aws_download: Optional[bool] = False
    category: str = "generic-web"
    every_token_is_sacred: bool = False

    def fill_empty_citation(self):
        if not self.citation:
            self.citation = f"https://huggingface.co/datasets/{self.address}"

    def fill_empty_license(self):
        if not self.license:
            self.license = "other"


class DataSources(BaseModel):
    sources: dict[str, DataSource]

    @classmethod
    def from_yaml(cls, file_path: str) -> "DataSources":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        sources = {name: DataSource(**source_data) for name, source_data in data.items()}
        for source in sources.values():
            source.fill_empty_citation()
            source.fill_empty_license()
        return cls(sources=sources)


def save_as_parquet(dataset, name, num_shards=128):
    dataset = dataset[list(dataset.keys())[0]]  # type: ignore[yolo]

    output_path_template = os.path.join(DOWNLOADED_DATASETS_PATH, name, "{index:05d}.parquet")
    for index in range(num_shards):
        if not Path(output_path_template.format(index=index)).exists():
            shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
            shard.to_parquet(output_path_template.format(index=index))


def download_datasets_raw():
    data_sources = DataSources.from_yaml("scripts/softwareheritage.yaml")

    for name, source in data_sources.sources.items():
        print("__________________________________________________________________________________________")
        print(f"Source: {name}")
        print(f"  Address: {source.address}")
        print(f"  Features: {', '.join(source.features)}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}")
        print()

        args = {
            "streaming": False,
            "cache_dir": CACHE_DIR,
            "trust_remote_code": True,
            "token": True,
            "num_proc": num_proc,
        }
        if source.requires_software_heritage_aws_download:
            url_dataset = datasets.load_dataset(source.address, **args)["train"]  # type: ignore
            dataset = software_heritage_aws_download(url_dataset, output_dir=os.path.join(CACHE_DIR, "the-stack-temp"))
            save_as_parquet(dataset, name)


if __name__ == "__main__":
    download_datasets_raw()
