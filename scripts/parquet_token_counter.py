"""Quick and dirty counter - need to refactor all of the data files at one point"""

import os
import glob
import math
import contextlib
import random

import itertools
import pyarrow as pa
import pyarrow.parquet as pq

from pydantic import BaseModel
from typing import List, Optional
import yaml


# current consts:
BASE_DIR = "/lustre/orion/csc569/scratch/jgeiping/data"
CACHE_DIR = f"{BASE_DIR}/test_cache"
FINAL_LOCATION = f"{BASE_DIR}/processed_dataset"
DATASET_STAGING = f"{BASE_DIR}/staging_dataset"

os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # cant make this work for now


import multiprocess

print(f"Number of CPUs available: {multiprocess.cpu_count()}")


import datasets

# from transformers import AutoTokenizer
from tokenizers import Tokenizer, Encoding

num_proc = 32  # normal andes node
target_block_size = 4096
target_tokenizer = "/lustre/orion/csc569/scratch/jgeiping/tokenizers/huginn_tokenizer_65k"
incinerate_ends = True  # False currently not implemented
target_shard_size_bytes = 2 * 1024**3
limiter = None  # limit to this many pq files for testing
resume = True
random.seed(1597)
log_tokenization_progress = True

if not log_tokenization_progress:
    datasets.disable_progress_bars()


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


def is_valid_parquet_file(file_path):
    # Parquet files end with a 4-byte magic number: 'PAR1'
    with open(file_path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        footer = f.read(4)
        if footer == b"PAR1":
            metadata = pq.read_metadata(file_path)
            schema = metadata.schema
            if len(schema) == 1 and schema.names == ["text"]:
                return True
        return False


def weighted_file_selection(files, weight):
    return files * math.floor(weight) + random.sample(files, k=round((weight % 1) * len(files)))


def _generate_tables(self, files):
    for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
        with open(file, "rb") as f:
            try:
                parquet_file = pq.ParquetFile(f)
                if parquet_file.metadata.num_row_groups > 0:
                    batch_size = self.config.batch_size or parquet_file.metadata.row_group(0).num_rows
                    for batch_idx, record_batch in enumerate(
                        parquet_file.iter_batches(batch_size=batch_size, columns=self.config.columns)
                    ):
                        pa_table = pa.Table.from_batches([record_batch])
                        yield f"{file_idx}_{batch_idx}", self._cast_table(pa_table)
            except Exception as e:
                print(f"Failed to read file '{file}' with error {type(e)}: {e}")


datasets.packaged_modules.parquet.Parquet._generate_tables = _generate_tables


def parquet_to_parquet_token_counter():
    tokenizer = Tokenizer.from_file(target_tokenizer + "/tokenizer.json")
    array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1"))
    array_task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    data_sources = DataSources.from_yaml("scripts/sources.yaml")
    if array_task_id > len(data_sources.sources):
        print(f"No work to do on worker {array_task_id}")
        return
    print(f"Working on task {array_task_id} out of {len(data_sources.sources)}")
    name, source = list(data_sources.sources.items())[array_task_id - 1]  # SLURM is 1-indexed

    files = glob.glob(f"{DATASET_STAGING}/{name}/**/*.parquet", recursive=True)[:limiter]
    valid_files = [p for p in files if is_valid_parquet_file(p)]
    parquet_files = weighted_file_selection(valid_files, source.weight)
    print(
        f"Gathered {len(files):<4} parquet sources from {name}, {len(valid_files):<4} were valid, "
        f"reweighing to {len(parquet_files):<4} with w={source.weight}."
    )

    print(f"Array task {array_task_id} out of {array_task_count} processing {len(parquet_files)} source files.")

    print(f"Discovered {len(parquet_files)} pre-staged parquet files. They better contain only 'text' fields.")
    datasets.utils._filelock.FileLock = contextlib.nullcontext
    monolith = datasets.load_dataset(
        "parquet",
        data_files=parquet_files,
        columns=["text"],
        streaming=False,
        cache_dir=CACHE_DIR,
        num_proc=num_proc,
        verification_mode="no_checks",
    )["train"]
    print(f"Monolith loaded with features {monolith.features}.")

    def tokenize_per_example(batch):
        encoded_batch: list[Encoding] = tokenizer.encode_batch(batch["text"])
        num_all_tokens = sum(len(ex.ids) for ex in encoded_batch)
        if source.every_token_is_sacred:
            num_tokens = num_all_tokens
        else:
            num_tokens = sum(len(ex.ids[: target_block_size + 1]) for ex in encoded_batch)
        doc_length_str = sum(len(t) for t in batch["text"]) / len(batch["text"])
        return {
            "num_valid_tokens": [num_tokens],
            "num_all_tokens": [num_all_tokens],
            "doc_length_str": [doc_length_str],
        }

    tokenized_monolith = monolith.map(
        tokenize_per_example,
        batch_size=1024,
        batched=True,
        num_proc=num_proc,
        remove_columns=monolith.column_names,
        keep_in_memory=True,
    )

    def merge_rows(batch):
        return {
            "num_valid_tokens_merged": [sum(ex for ex in batch["num_valid_tokens"])],
            "num_all_tokens_merged": [sum(ex for ex in batch["num_all_tokens"])],
            "doc_length_str_merged": [sum(ex for ex in batch["doc_length_str"]) / len(batch["doc_length_str"])],
        }

    merged_counter = tokenized_monolith.map(
        merge_rows,
        batch_size=1024,
        batched=True,
        num_proc=num_proc,
        remove_columns=tokenized_monolith.column_names,
        keep_in_memory=True,
    )

    def merge_rows(batch):
        return {
            "num_valid_tokens_merged_full": [sum(ex for ex in batch["num_valid_tokens_merged"])],
            "num_all_tokens_merged_full": [sum(ex for ex in batch["num_all_tokens_merged"])],
            "doc_length_str_merged_full": [
                sum(ex for ex in batch["doc_length_str_merged"]) / len(batch["doc_length_str_merged"])
            ],
        }

    merged_counter = merged_counter.map(
        merge_rows,
        batch_size=len(merged_counter),
        batched=True,
        remove_columns=merged_counter.column_names,
        keep_in_memory=True,
    )
    assert len(merged_counter) == 1
    count = merged_counter["num_valid_tokens_merged_full"][0]
    total_tokens = merged_counter["num_all_tokens_merged_full"][0]
    avg_doc_length = round(merged_counter["doc_length_str_merged_full"][0])
    print(
        f"Dataset {name} counted. Num chosen tokens: {count/1e9:7.4f}B "
        f"out of {total_tokens/1e9:7.4f}B  total tokens.\n"
        f"{(total_tokens - count)/1e9:7.4f}B tokens incinerated ( {(total_tokens - count)/total_tokens:%} )"
        f"due to average document length of {avg_doc_length} characters in this source."
    )
    print(f"--->{count}--{total_tokens}--{avg_doc_length}")


if __name__ == "__main__":
    parquet_to_parquet_token_counter()
