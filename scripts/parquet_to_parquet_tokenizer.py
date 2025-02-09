"""Another data format :>


Problems with recpre pkds: - Writer is hard to read, and I don't understand the sep token problem
Problem with datatrove format - A single tokenized file will be a drama on frontier
Also, neither seem to be able to give me the packing control I want.

New Strategy:
- Phase I: Tokenizer parquet -> parquet via Huggingface datasets functionality (and patience)
- Phase II: Full shuffle based on datasets shuffler
- Phase III: Map to blocks (of size S+1), measure incineration rate.
- Phase IV: Write back into parquet

Later: Read as HF dataset with parquet contents.


Parallelizes across nodes with each node picking up %nodes file from the parquet files glob.
Should launch 1 task per node.
"""

import os
import glob
import math
from pathlib import Path
import contextlib
import random

import itertools
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.parquet as pq

from pydantic import BaseModel
from typing import List, Optional
import yaml


# current consts:
BASE_DIR = "/lustre/orion/csc569/proj-shared/language_datasets"
CACHE_DIR = f"{BASE_DIR}/cache"
FINAL_LOCATION = f"{BASE_DIR}/processed"
DATASET_STAGING = f"{BASE_DIR}/staging_dataset"
final_dataset_name = "recllm_project_cooldown_v0_pqds"

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

shuffle_filenames = True
limiter = None  # limit to this many pq files for testing
resume = False
random.seed(1597)
log_tokenization_progress = True
local_cache = False  # no burst buffer on andes

if local_cache:
    CACHE_DIR = "/mnt/bb/jgeiping/tmp"

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
            else:
                pass
                # print(schema)
                # raise ValueError()
        return False


def weighted_file_selection(files, weight):
    return files * math.floor(weight) + random.sample(files, k=round((weight % 1) * len(files)))


# These are previous attempts to speed up the parquet-> arrow conversion
# add yours here at your own risk
def generate_tables(self, files):
    for file_idx, file in enumerate(itertools.chain.from_iterable(files)):
        table = pq.read_table(file, columns=self.config.columns)
        yield f"{file_idx}", self._cast_table(table)


def generate_tables_ds(self, files):
    flat_files = list(itertools.chain.from_iterable(files))
    dataset = ds.dataset(flat_files, format="parquet")
    table = dataset.to_table()
    yield "Using arrow's internal dataset framework", self._cast_table(table)


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


def process_shards(dataset, num_shards, output_template, rank=None, size=None):
    if resume:
        unfinished_shards = [
            index for index in range(num_shards) if not Path(output_template.format(rank=rank, index=index)).exists()
        ]
    else:
        unfinished_shards = range(num_shards)

    for index in unfinished_shards:
        output_path = output_template.format(rank=rank, index=index)
        if not (Path(output_path).exists() and resume):
            shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
            shard.to_parquet(output_path)
            if rank is not None:
                print(f"Rank {rank}/{size}: Saved shard {index + 1}/{num_shards}.")
            else:
                print(f"Saved shard {index + 1}/{num_shards}")


def print_examples(array_task_id):
    tokenizer = Tokenizer.from_file(target_tokenizer + "/tokenizer.json")
    # Print some rows from the 1st file
    example = os.path.join(FINAL_LOCATION, final_dataset_name, "train", f"{array_task_id:03d}_{1:05d}.parquet")
    parquet_file = pq.ParquetFile(example)
    print("Schema:")
    print(parquet_file.schema)

    first_batch = next(parquet_file.iter_batches(batch_size=5))
    print("\nFirst 20 entries from first 5 rows:")
    values = first_batch["input_ids"].to_pylist()
    for seq in values:
        print(seq[:20])

    print("\nDecoded text from row 2:")
    print(tokenizer.decode(first_batch["input_ids"][2].as_py(), skip_special_tokens=False))
    begin_text_id = tokenizer.encode("<|begin_text|>", add_special_tokens=False).ids[0]
    begin_header_id = tokenizer.encode("<|begin_header|>", add_special_tokens=False).ids[0]

    def count_special_tokens(parquet_path):
        parquet_file = pq.ParquetFile(parquet_path)
        num_sequences = parquet_file.metadata.num_rows
        begin_text_count = 0
        begin_header_count = 0
        for batch in parquet_file.iter_batches():
            for input_ids in batch["input_ids"].to_pylist():
                begin_text_count += input_ids.count(begin_text_id)
                begin_header_count += input_ids.count(begin_header_id)
        return begin_text_count, begin_header_count, num_sequences

    begin_text, begin_header, num_sequences = count_special_tokens(example)
    print(f"\nAnalysis of {num_sequences} sequences:")
    print(f"<|begin_text|>: {begin_text} total ({begin_text / num_sequences:.2f}  per sequence)")
    print(f"<|begin_header|>: {begin_header} total ({begin_header / num_sequences:.2f} per sequence)")


def parquet_to_parquet_tokenization():
    tokenizer = Tokenizer.from_file(target_tokenizer + "/tokenizer.json")

    #
    # - Phase I: Tokenizer parquet -> parquet via Huggingface datasets functionality (and patience) ####################
    #

    parquet_files = []
    data_sources = DataSources.from_yaml("scripts/cooldown_set_v0.yaml")
    source_reference = {}

    for name, source in data_sources.sources.items():
        try:
            files = glob.glob(f"{DATASET_STAGING}/{name}/**/*.parquet", recursive=True)[:limiter]
        except Exception:
            files = []
        valid_files = [p for p in files if is_valid_parquet_file(p)]
        weighed_files = weighted_file_selection(valid_files, source.weight)
        print(
            f"Gathered {len(files):<4} parquet sources from {name:<30}, {len(valid_files):<4} were valid, "
            f"reweighing to {len(weighed_files):<4} with w={source.weight}."
        )
        parquet_files += weighed_files
        for source_file in weighed_files:
            source_reference[source_file] = name

    array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1")) - 1
    array_task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    if shuffle_filenames:
        random.seed(233)
        random.shuffle(parquet_files)
    parquet_files = parquet_files[array_task_id::array_task_count]

    print(f"Array task {array_task_id} out of {array_task_count} processing {len(parquet_files)} source files.")

    print(f"Discovered {len(parquet_files)} pre-staged parquet files. They better contain only 'text' fields.")
    datasets.utils._filelock.FileLock = contextlib.nullcontext

    # Bulk load like before
    monolith = datasets.load_dataset(
        "parquet",
        data_files=parquet_files,
        columns=["text"],
        streaming=False,
        cache_dir=CACHE_DIR,
        num_proc=num_proc,
        verification_mode="no_checks",
    )["train"]

    # Get all sizes
    file_sizes = []
    for f in parquet_files:
        try:
            size = pq.read_metadata(f).num_rows
            file_sizes.append(size)
        except Exception as e:
            raise ValueError(f"Cannot read size of {f}: {e}")

    # Create flags
    keep_all_values = []
    for file, size in zip(parquet_files, file_sizes):
        value = bool(data_sources.sources[source_reference[file]].every_token_is_sacred)
        keep_all_values.extend([value] * size)

    # Strict verification
    if len(keep_all_values) != len(monolith):
        raise ValueError(
            f"Size mismatch: flags={len(keep_all_values)} vs dataset={len(monolith)}\n"
            f"File sizes: {list(zip(parquet_files, file_sizes))}"
        )

    monolith = monolith.add_column("keep_all", keep_all_values)
    # Add right after loading monolith:
    print(f"Dataset loaded with {len(monolith)} rows")
    print(f"File sizes: {list(zip(parquet_files, file_sizes))}")
    print(f"Total size from files: {sum(file_sizes)}")

    # After adding keep_all column:
    true_count = sum(1 for x in keep_all_values if x)
    print(f"Keep_all distribution: {true_count} True, {len(keep_all_values) - true_count} False")
    # Add after everything:
    for i in range(min(5, len(monolith))):
        print(f"Row {i}: text={monolith[i]['text']}")
        print(f"Row {i}: keep_all={monolith[i]['keep_all']}")

    print(f"Monolith loaded with features {monolith.features}.")

    def tokenize_per_example(batch):
        # input_batch = []
        # for text, keep_all in zip(batch["text"], batch["keep_all"]):
        #     if keep_all:
        #         input_batch += [text]
        #     else:
        #         input_batch += [text[: (16 * target_block_size)]]
        # encoded_batch: list[Encoding] = tokenizer.encode_batch(input_batch)
        # output_batch = []
        # for encoding, keep_all in zip(encoded_batch, batch["keep_all"]):
        #     if keep_all:
        #         output_batch += [encoding.ids]
        #     else:
        #         output_batch += [encoding.ids[: target_block_size + 64]]
        # return {"token_ids": output_batch}
        encoded_batch: list[Encoding] = tokenizer.encode_batch(batch["text"])
        return {"token_ids": [ex.ids for ex in encoded_batch]}

    tokenized_monolith = monolith.map(
        tokenize_per_example, batch_size=1024, batched=True, num_proc=num_proc, remove_columns=["text"]
    )
    print(f"Monolith tokenized with features {tokenized_monolith.features}.")

    #
    # - Phase II: Full shuffle based on datasets shuffler ##############################################################
    #

    tokenized_monolith = tokenized_monolith.shuffle(seed=777777777777)

    #
    # - Phase III: Map to blocks (of size S+1), measure incineration rate. #############################################
    #

    def map_to_block_operation(batch):
        """Desirable properties: Only block starts are used, blocks are filled from random rows from all datasets."""
        mined_blocks = []
        # incinerated_tokens = # should probably count this later, but we can estimate from loss in number of tokens
        # Concatenate batch contents:
        current_block = []
        last_block_had_keep_all = False
        for example, keep_all in zip(batch["token_ids"], batch["keep_all"]):
            if len(current_block) >= target_block_size + 1:
                if last_block_had_keep_all:
                    while len(current_block) >= target_block_size + 1:
                        mined_blocks.append(current_block[: (target_block_size + 1)])
                        current_block = current_block[(target_block_size + 1) :]
                else:
                    mined_blocks.append(current_block[: (target_block_size + 1)])
                    current_block = []
            current_block += example
            last_block_had_keep_all = keep_all
        return {"input_ids": mined_blocks}

    blockset = tokenized_monolith.map(
        map_to_block_operation,
        batch_size=4096,
        batched=True,
        num_proc=num_proc,
        remove_columns=tokenized_monolith.column_names,
    )
    print(f"Blockset written with features {blockset.features}.")
    num_blocks = len(blockset)
    print(f"Number of valid blocks: {num_blocks}")
    print(f"This is equivalent to {num_blocks * (target_block_size + 1) / 1e9:7.4f}B tokens.")
    print(
        f"Total token count is likely around {array_task_count * num_blocks * (target_block_size + 1) / 1e12:7.4f}T tokens"
    )
    #
    # - Phase IV: Write back into parquet ##############################################################################
    #

    # num_shards = math.ceil(num_blocks * (target_block_size + 1) / target_shard_size_bytes)
    num_shards = math.ceil(4096 / array_task_count)  # 4096 shards is enough for 512 * 8 ranks
    # print(f"Preparing for {num_shards} shards of size {target_shard_size_bytes}")
    print(f"Preparing for {num_shards} shards.")

    blockdict = blockset.train_test_split(train_size=0.95, shuffle=False)

    output_template_train = os.path.join(FINAL_LOCATION, final_dataset_name, "train", "{rank:03d}_{index:05d}.parquet")
    output_template_val = os.path.join(FINAL_LOCATION, final_dataset_name, "val", "{rank:03d}_{index:05d}.parquet")
    process_shards(blockdict["train"], num_shards, output_template_train, rank=array_task_id, size=array_task_count)
    process_shards(blockdict["test"], num_shards, output_template_val, rank=array_task_id, size=array_task_count)
    print(f"Reshuffling into parquet finished on task {array_task_id}!")
    print("------------------------------------------------------------------------------")
    print("-----------------------------Printing examples now ---------------------------")
    print_examples(array_task_id)


if __name__ == "__main__":
    parquet_to_parquet_tokenization()
