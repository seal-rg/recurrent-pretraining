"""Data gen script for OLCF pretraing

Assumptions:
1) Tokenizer is already generated and HF-compatible
2) Dataset component are in .yaml format as in tokenizer_generation.py
3) No data reweighing will take place at runtime.
4) Sequence length is fixed and known.
5) We will pack into the old recpre format.

This is just the downloader though, as andes nodes are note connected to the internet.

"""

from pydantic import BaseModel
from typing import List, Optional, TYPE_CHECKING
import yaml
import os
from datetime import datetime
import subprocess
import time

from pathlib import Path
import json
import io
import contextlib
import boto3
import smart_open

from botocore import UNSIGNED
from botocore.config import Config


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # cant make this work for now


# from transformers import PreTrainedTokenizerFast
from huggingface_hub import snapshot_download


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


num_proc = 64
datasets.config.STREAMING_READ_MAX_RETRIES = 50
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10


# monkey-patch of zstd to fix bad formatting in proof-pile:
import zstandard as zstd


@contextlib.contextmanager
def safe_open(file, *args, **kwargs):
    if isinstance(file, io.IOBase):
        # If it's already a file object, yield it
        yield file
    else:
        try:
            # Try to open as a zstd file
            with zstd.open(file, *args, **kwargs) as f:
                yield f
        except zstd.ZstdError:
            # If that fails, open as a regular file
            with open(file, *args, **kwargs) as f:
                yield f


# Replace zstd.open with safe_open patch
zstd.open = safe_open


class DataSource(BaseModel):
    address: str
    subset: Optional[str] = None
    features: List[str]
    needs_chat_templating: bool = False
    license: Optional[str] = None
    citation: Optional[str] = None
    machine_generated: bool = False
    requires_software_heritage_aws_download: Optional[bool] = False
    weight: float = 1.0
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


def software_heritage_aws_download(dataset):
    def download_contents(files):
        s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        contents = []
        for repo_file in files:
            s3_url = f"s3://softwareheritage/content/{repo_file['blob_id']}"

            with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                # contents.append({"file": repo_file["path"], "text": fin.read().decode(repo_file["src_encoding"])})
                contents.append(fin.read().decode(repo_file["src_encoding"]))

        return {"content": "\n".join(contents)}

    dataset = dataset.map(lambda row: download_contents(row["files"]), num_proc=num_proc)
    return dataset


from concurrent.futures import ThreadPoolExecutor


def software_heritage_aws_download_threadpool(dataset, batch_size=1024, num_proc=64):
    s3_config = Config(signature_version="UNSIGNED", max_pool_connections=50)

    def download_file(repo_file):
        s3 = boto3.client("s3", config=s3_config)
        s3_url = f"s3://softwareheritage/content/{repo_file['blob_id']}"
        with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
            return fin.read().decode(repo_file["src_encoding"])

    def process_batch(batch):
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            contents = list(executor.map(download_file, batch["files"]))

        # Combine all file contents for each example in the batch
        return {
            "content": ["\n".join(contents[i : i + len(example["files"])]) for i, example in enumerate(batch["files"])]
        }

    return dataset.map(
        process_batch, batched=True, batch_size=batch_size, num_proc=num_proc, remove_columns=dataset.column_names
    )


def generator_from_json(jsonl_files):
    def generator():
        for file_path in jsonl_files:
            with open(file_path, "r") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if "text" in data:
                            yield {"text": data["text"]}
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON in file: {file_path}")
                        continue

    return generator


def custom_mathpile_download():
    mathpile_dir = Path(DOWNLOADED_DATASETS_PATH) / "MathPile"
    mathpile_dir.mkdir(parents=True, exist_ok=True)

    # Download MathPile dataset
    subprocess.run(
        [
            "huggingface-cli",
            "download",
            "--repo-type",
            "dataset",
            "GAIR/MathPile",
            "--local-dir",
            str(mathpile_dir),
        ],
        check=True,
    )
    # Decompress .gz files
    gz_files = list(mathpile_dir.rglob("*.gz"))
    decompressed_count = 0
    skipped_count = 0
    for gz_file in gz_files:
        jsonl_file = gz_file.with_suffix("")
        if not jsonl_file.exists():
            subprocess.run(["gzip", "-d", "-f", str(gz_file)], check=True)
            decompressed_count += 1
        else:
            skipped_count += 1
    print(f"Decompressed {decompressed_count} .gz files, skipped {skipped_count} already decompressed files")

    # Find all .jsonl files in the train directory and its subdirectories
    train_dir = mathpile_dir / "train"
    jsonl_files = list(train_dir.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} .jsonl files in {train_dir}")


def save_as_parquet(dataset, name, num_shards=128):
    dataset = dataset[list(dataset.keys())[0]]  # type: ignore[yolo]

    output_path_template = os.path.join(DOWNLOADED_DATASETS_PATH, name, "{index:05d}.parquet")
    for index in range(num_shards):
        if not Path(output_path_template.format(index=index)).exists():
            shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
            shard.to_parquet(output_path_template.format(index=index))


def download_datasets_raw():
    data_sources = DataSources.from_yaml("scripts/sources.yaml")

    for name, source in data_sources.sources.items():
        print("__________________________________________________________________________________________")
        print(f"Source: {name}")
        print(f"  Address: {source.address}")
        print(f"  Features: {', '.join(source.features)}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}")
        print()

        args = {"streaming": False, "cache_dir": CACHE_DIR, "trust_remote_code": True, "token": True, "num_proc": 32}
        if source.subset:
            args |= {"name": source.subset}
        if "MathPile" in name:  # special rules for mathpile...
            custom_mathpile_download()
        elif "dolma" in name:
            from datasets.utils.file_utils import get_datasets_user_agent
            from datasets.download.download_config import DownloadConfig

            download_config = DownloadConfig(
                max_retries=5,  # Increase the number of retries
                num_proc=32,
                user_agent=get_datasets_user_agent(),  # Set a custom user agent
                force_download=False,  # Don't re-download if the file is already cached
                resume_download=True,  # Resume download if an incomplete file is found
                disable_tqdm=False,
            )
            dataset = datasets.load_dataset("recpre/data/dolma.py", download_config=download_config, **args)
            save_as_parquet(dataset, name)
        elif "redpajama" in name:
            from datasets.utils.file_utils import get_datasets_user_agent
            from datasets.download.download_config import DownloadConfig

            download_config = DownloadConfig(
                max_retries=5,  # Increase the number of retries
                num_proc=32,
                user_agent=get_datasets_user_agent(),  # Set a custom user agent
                force_download=False,  # Don't re-download if the file is already cached
                resume_download=True,  # Resume download if an incomplete file is found
                disable_tqdm=False,
            )
            dataset = datasets.load_dataset("recpre/data/redpajama_1T.py", download_config=download_config, **args)
            save_as_parquet(dataset, name)
        elif source.requires_software_heritage_aws_download:
            url_dataset = datasets.load_dataset(source.address, **args)
            dataset = software_heritage_aws_download_threadpool(url_dataset)  # data is materialized here!
            save_as_parquet(dataset, name)
        else:
            for retry in range(20):
                try:
                    snapshot_download(
                        source.address,
                        repo_type="dataset",
                        local_dir=os.path.join(DOWNLOADED_DATASETS_PATH, name),
                        max_workers=num_proc,
                    )
                    break
                except Exception as e:  # (HTTPError, ConnectionError, ChunkedEncodingError):
                    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")  # type: ignore
                    time.sleep(7)
                    print(f"Starting retry number {retry} ...")


if __name__ == "__main__":
    download_datasets_raw()
