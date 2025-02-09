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
import pyarrow as pa
import pyarrow.parquet as pq


shuffle_round = 0

# current consts:
BASE_DIR = "/lustre/orion/csc569/proj-shared/language_datasets"
CACHE_DIR = f"{BASE_DIR}/cache"
FINAL_LOCATION = f"{BASE_DIR}/processed"
if shuffle_round == 0:
    original_dataset_name = "recllm_project_v02_pqds"
else:
    original_dataset_name = f"recllm_project_v02_pqds_reshuffled_{shuffle_round - 1}"
final_dataset_name = f"recllm_project_v02_pqds_reshuffled_{shuffle_round}"

os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import multiprocess

print(f"Number of CPUs available: {multiprocess.cpu_count()}")


import datasets

# from transformers import AutoTokenizer
from tokenizers import Tokenizer

num_proc = 32  # normal andes node
target_block_size = 4096
target_tokenizer = "/lustre/orion/csc569/scratch/jgeiping/tokenizers/huginn_tokenizer_65k"

target_shard_size_bytes = 1024**3 // 2  # optimal would be 2 *1024**3, but we need more to split across 4096 ranks

shuffle_filenames = True
resume = False
random.seed(1597 * (shuffle_round + 1) % (2**31 - 1))
log_tokenization_progress = True

if not log_tokenization_progress:
    datasets.disable_progress_bars()


# These are previous attempts to speed up the parquet-> arrow conversion
# add yours here at your own risk
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
    #
    # - Phase I: Reload a pseudorandom chunk of parquet files on this worker
    #
    parquet_files = glob.glob(f"{FINAL_LOCATION}/{original_dataset_name}/**/*.parquet", recursive=True)

    array_task_id = int(os.getenv("SLURM_ARRAY_TASK_ID", "1")) - 1
    array_task_count = int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    if shuffle_filenames:
        random.seed(501235938 * (shuffle_round + 1) % (2**31 - 1))
        random.shuffle(parquet_files)
    parquet_files = parquet_files[array_task_id::array_task_count]

    print(f"Array task {array_task_id} out of {array_task_count} processing {len(parquet_files)} source files.")
    datasets.utils._filelock.FileLock = contextlib.nullcontext
    print(parquet_files)
    # Bulk load like before
    blockset = datasets.load_dataset(
        "parquet",
        data_files=parquet_files,
        columns=["input_ids"],
        streaming=False,
        cache_dir=CACHE_DIR,
        num_proc=num_proc,
        # verification_mode="no_checks",
    )["train"]
    #
    # - Phase II: Full shuffle based on datasets shuffler ##############################################################
    #
    blockset = blockset.shuffle(seed=7777777777 * (shuffle_round + 1) % (2**31 - 1))
    num_blocks = len(blockset)
    print(f"Number of valid blocks: {num_blocks}")
    print(f"This is equivalent to {num_blocks * (target_block_size + 1) / 1e9:7.4f}B tokens.")
    print(
        f"Total token count is likely around {array_task_count * num_blocks * (target_block_size + 1) / 1e12:7.4f}T tokens"
    )
    #
    # - Phase III: Write back into parquet ##############################################################################
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
