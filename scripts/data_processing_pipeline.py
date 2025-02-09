"""Data gen script for OLCF pretraing

Assumptions:
1) Tokenizer is already generated and HF-compatible
2) Dataset component are in .yaml format as in tokenizer_generation.py
3) No data reweighing will take place at runtime.
4) Sequence length is fixed and known.
5) We will pack into the old recpre format.

The first run will be minimal processing, implying
1) No global deduplication, re-occurences of snippets in multiple sources is a feature
2) Tail ends of documents will be cut off.
3) Language filtering on?
4) Tails are cut out, short sequences are allowed (maybe we'll pack into few-shot later),
    loss masks can be made on the fly, no need to make them here



Steps planned for this file:
1) Instantiate dataset object based on downloaded stuff
2) Select desired subsets
3) Apply chat format if necessary
4) Convert to parquet blobs that can be read with datatrove
5) ftfy + lang_filter through datatrove?
6) Compute token counts per pile

"""

import multiprocessing

# Set the start method to 'spawn' or 'forkserver'
multiprocessing.set_start_method("forkserver", force=True)


from pydantic import BaseModel
from typing import List, Optional, TYPE_CHECKING
import yaml
import os
from datetime import datetime

from pathlib import Path
import json
import io
import contextlib

import math
from functools import partial


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # cant make this work for now


# from transformers import PreTrainedTokenizerFast


# tokenizer = PreTrainedTokenizerFast("tomg-group-umd/huginn_tokenizer_test")
# tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|begin_header|>' + message['role'] + '<|end_header|>\n\n'+ message['content'] | trim + '<|end_turn|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% endif %}"

# current consts:
BASE_DIR = "/lustre/orion/csc569/scratch/jgeiping/data"
DOWNLOADED_DATASETS_PATH = f"{BASE_DIR}/test_download_folder"  # /test_download_folder

# new base dir to move onto shared drives
BASE_DIR = "/lustre/orion/csc569/proj-shared/language_datasets"
CACHE_DIR = f"{BASE_DIR}/cache"
FINAL_LOCATION = f"{BASE_DIR}/processed_dataset"
DATASET_STAGING = f"{BASE_DIR}/staging_dataset"
resume = True  # only write new files

os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# Create a no-op FileLock that mimics the structure in datasets.utils._filelock
class NoOpFileLock:
    MAX_FILENAME_LENGTH = 255

    def __init__(self, lock_file, *args, **kwargs):
        pass  # print(f"NoOpFileLock initialized with {lock_file}")

    def __enter__(self):
        # print("NoOpFileLock entered")
        return self

    def __exit__(self, *args):
        pass  # print("NoOpFileLock exited")

    def acquire(self, *args, **kwargs):
        pass  # print("NoOpFileLock acquire called")

    def release(self, *args, **kwargs):
        pass  # print("NoOpFileLock release called")

    @classmethod
    def hash_filename_if_too_long(cls, path: str) -> str:
        # print(f"hash_filename_if_too_long called with {path}")
        return path


# Also replace it in the filelock library itself, in case it's imported directly
import filelock

filelock.FileLock = NoOpFileLock

import datasets
# import huggingface_hub

# somehow required otherwise the token is not always read correctly:
# huggingface_hub.login(token=os.environ.get("MANUALLY_SET_HF_TOKEN", None), add_to_git_credential=True)

if TYPE_CHECKING:
    import datasets.config


num_proc = 64
datasets.config.STREAMING_READ_MAX_RETRIES = 50
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10

# Monkey-patch the FileLock in datasets to use a no-op context manager
datasets.utils._filelock.FileLock = contextlib.nullcontext
import contextlib


# Patch the lock_file function
def no_op_lock_file(*args, **kwargs):
    print("no_op_lock_file called")
    return contextlib.nullcontext()


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


def get_mathpile():
    mathpile_dir = Path(DOWNLOADED_DATASETS_PATH) / "MathPile"
    # Find all .jsonl files in the train directory and its subdirectories
    train_dir = mathpile_dir / "train"
    jsonl_files = list(train_dir.rglob("*.jsonl"))
    print(f"Mathpile: Found {len(jsonl_files)} .jsonl files in {train_dir}")

    # Create a streaming dataset directly from local JSONL files
    dataset = datasets.Dataset.from_generator(generator_from_json(jsonl_files))
    return dataset


def load_dataset_without_default_split(source, args):
    print("Attempting to fix by detecting and concatenating all subsets.")
    configs = datasets.get_dataset_config_names(source)
    splits = []
    for config in configs:
        print(f"Loading {config}")
        subset = datasets.load_dataset(source, config, **args)
        if isinstance(subset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            subset = subset[list(subset.keys())[0]]
        splits.append(subset)
    dataset = datasets.concatenate_datasets(splits)
    return dataset


def save_shard(dataset, output_path_template, name, index, num_shards):
    output_path = output_path_template.format(index=index)
    if not (Path(output_path).exists() and resume):
        shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
        shard.to_parquet(output_path)
        print(f"Saved shard {index + 1}/{num_shards} of {name}")


def save_as_parquet(dataset, name, target_shard_size_bytes=2 * 1024 * 1024 * 1024, num_proc=32):
    # Calculate number of shards
    if dataset.info.size_in_bytes is not None:
        print(f"Dataset size according to HF: {dataset.info.size_in_bytes / (1024**3):.2f} GB")
        num_shards = math.ceil(dataset.info.size_in_bytes / target_shard_size_bytes)
    else:
        num_shards = 1024
    print(f"Target shard size: {target_shard_size_bytes / (1024**3):.2f} GB")
    print(f"Number of shards: {num_shards}")

    output_path_template = os.path.join(DATASET_STAGING, name, "{index:05d}.parquet")
    if resume:
        unfinished_shards = [
            index for index in range(num_shards) if not Path(output_path_template.format(index=index)).exists()
        ]
    else:
        unfinished_shards = range(num_shards)
    save_shard_partial = partial(save_shard, dataset, output_path_template, name, num_shards=num_shards)

    if num_proc > 0:
        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_proc) as pool:
            pool.map(save_shard_partial, unfinished_shards)
    else:
        for shard in unfinished_shards:
            save_shard_partial(shard)


def check_if_shards_are_written(dataset, name, target_shard_size_bytes=2 * 1024 * 1024 * 1024, num_proc=32):
    if dataset.info.size_in_bytes is not None:
        num_shards = math.ceil(dataset.info.size_in_bytes / target_shard_size_bytes)
    else:
        num_shards = 1024
    output_path_template = os.path.join(DATASET_STAGING, name, "{index:05d}.parquet")
    unfinished_shards = [
        index for index in range(num_shards) if not Path(output_path_template.format(index=index)).exists()
    ]
    return len(unfinished_shards) == 0


names_for_roles = ["role", "from"]
names_for_content = ["content", "value"]
names_for_entities = ["gpt", "assistant"]


def get_unifying_formatter(features, tokenizer, needs_chat_templating=False):
    def process_example(example):
        if len(features) > 1:
            if needs_chat_templating:
                output = [
                    {"role": "user" if i == 0 else "assistant", "content": str(example[feat])}
                    for i, feat in enumerate(features)
                    if feat in example
                ]

            else:
                output = "\n".join(str(example[feat]) for feat in features if feat in example)
        else:
            if needs_chat_templating:
                output = [
                    {
                        "role": next((m.get(k) for k in names_for_roles if k in m), None),
                        "content": next((m.get(k) for k in names_for_content if k in m), None),
                    }
                    for m in example[features[0]]
                ]
            else:
                output = str(example[features[0]]) if features[0] in example else ""

        if needs_chat_templating:
            # example["messages"] = output # let's give up on two-column formats with huggingface datasets
            for m in output:
                if m["role"] in names_for_entities:  # type: ignore
                    m["role"] = "Huginn"  # type: ignore
            text = tokenizer.apply_chat_template(output, tokenize=False, add_generation_prompt=False)
        else:
            text = output
        return {"text": text}

    # Combine all processing steps into a single map operation
    # remove_columns = [c for c in dataset.column_names if c != "text"]
    # result = dataset.map(lambda row: process_example(row), remove_columns=remove_columns, num_proc=num_proc)
    return process_example


# type: ignore
from datatrove.utils.lid import FT176LID  # , GlotLID
from datatrove.data import Document

# List of languages to filter (excluding English)
target_languages = [
    "zh",  # Chinese
    "es",  # Spanish
    "ar",  # Arabic
    "pt",  # Portuguese
    "id",  # Indonesian
    "fr",  # French
    "ja",  # Japanese
    "ru",  # Russian
    "de",  # German
    "ko",  # Korean
    "tr",  # Turkish
    "it",  # Italian
    "pl",  # Polish
    "nl",  # Dutch
    "hi",  # Hindi
    "fa",  # Persian
    "vi",  # Vietnamese
    "th",  # Thai
    "sv",  # Swedish
    "he",  # Hebrew
    "cs",  # Czech
    "ro",  # Romanian
    "el",  # Greek
    "da",  # Danish
    "fi",  # Finnish
    "no",  # Norwegian
    "hu",  # Hungarian
    "bn",  # Bengali
    "uk",  # Ukrainian
    "ms",  # Malay
]
model = FT176LID(target_languages + ["English"])


def language_filter(example, non_english_threshold=0.5):
    try:
        doc = Document(
            text=example["text"]
            if "text" in example and example["text"] is not None
            else "\n".join(m["content"] for m in example["messages"] if m["content"] is not None)
            if "messages" in example and len(example["messages"]) > 0
            else " ",
            id="f123",
        )
    except Exception as e:
        print(e)
        return False
    best_lang_pair, lang_pairs = model.predict(doc)
    lang, lang_score = best_lang_pair
    if lang in target_languages and lang_score > non_english_threshold:
        print("------------------------------------------------------------------------------------------------------")
        print(f"Filtered: {lang} ({lang_score:.2f})")
        print(doc.text)
        return False
    return True


import ftfy
from transformers import AutoTokenizer


def format_ftfy(texts: str):
    return {"text": [ftfy.fix_text(text) for text in texts]}


def filter_json(file_path: str):
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            text = data.get("text", "")
            if len(text) > 0:
                yield {"text": text}


import glob
import os


def get_file_features(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline()
        return set(json.loads(first_line).keys())


def select_matrix_files(base_path):
    all_files = glob.glob(os.path.join(base_path, "book_*.jsonl"))
    file_groups = {}

    for file in all_files:
        features = frozenset(get_file_features(file))
        if features not in file_groups:
            file_groups[features] = []
        file_groups[features].append(file)

    return file_groups


from datasets import concatenate_datasets


def load_and_concatenate_jsonl_files(name, source, subset, base_path):
    # Get all JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(base_path, f"{subset}_*.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {base_path}")

    print(f"Found {len(jsonl_files)} JSONL files")

    # Load each file separately
    datasets_list = []
    for file in jsonl_files:
        try:
            dataset = datasets.load_dataset("json", data_files=file, cache_dir=CACHE_DIR, num_proc=num_proc)
            if isinstance(dataset, datasets.DatasetDict):
                if "train" in dataset:
                    dataset = dataset["train"]
                else:
                    print(f"No 'train' split in {file}. Available splits: {dataset.keys()}")
                    continue

            # Keep only the 'text' column
            if "text" in dataset.column_names:
                dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])
                datasets_list.append(dataset)
        except Exception as e:
            print(f"Error loading file {file}: {str(e)}")

    return concatenate_datasets(datasets_list)


@contextlib.contextmanager
def safe_open(file, *args, **kwargs):
    if isinstance(file, io.IOBase):
        yield file
    else:
        with open(file, "rb") as f:
            zstd_reader = zstd.ZstdDecompressor().stream_reader(f)
            yield io.TextIOWrapper(zstd_reader, encoding="utf-8")


def process_files(files):
    for file in files:
        try:
            with safe_open(file, "rt") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if text:
                            yield {"text": text}
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")


import datasets
from datasets import Features, Value


# Define the full schema based on the error message
automath1_features = Features(
    {
        "url": Value("string"),
        "text": Value("string"),
        "date": Value("timestamp[s]"),
        "meta": {
            "domain": Value("string"),
            "url": Value("string"),
            "openwebmath_score": Value("float64"),
            "openwebmath_perplexity": Value("float64"),
            "lm_name": Value("string"),
            "lm_label": Value("string"),
            "lm_q1_score": Value("float64"),
            "lm_q2_score": Value("float64"),
            "lm_q1q2_score": Value("float64"),
        },
    }
)


automath2_features = Features(
    {
        "url": Value("string"),
        "title": Value("string"),
        "abstract": Value("string"),
        "text": Value("string"),
        "meta": {
            "timestamp": Value("timestamp[s]"),
            "yymm": Value("string"),
            "arxiv_id": Value("string"),
            "language": Value("string"),
            "url": Value("string"),
            "abstract": Value("string"),
            "subjects": Value("string"),
            "title": Value("string"),
            "lm_name": Value("string"),
            "lm_label": Value("string"),
            "lm_q1_score": Value("float64"),
            "lm_q2_score": Value("float64"),
            "lm_q1q2_score": Value("float64"),
        },
        "date": Value("timestamp[s]"),
    }
)


def load_json_only(data_files):
    def process_files(files):
        for file in files:
            try:
                with open(file, "rt") as f:
                    for line in f:
                        data = json.loads(line)
                        text = data.get("text", "").strip()
                        if text:
                            yield {"text": text}
            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")

    return datasets.Dataset.from_generator(
        process_files,
        features=Features({"text": Value("string")}),
        gen_kwargs={"files": data_files},
        cache_dir=CACHE_DIR,
        num_proc=num_proc,
    )


def load_text_only(data_files):
    return datasets.Dataset.from_generator(
        process_files,
        features=Features({"text": Value("string")}),
        gen_kwargs={"files": data_files},
        cache_dir=CACHE_DIR,
        num_proc=num_proc,
    )


def verify_dataset_can_be_loaded_and_process(filter_language=False, run_ftfy=False):
    data_sources = DataSources.from_yaml("scripts/sources.yaml")
    tokenizer = AutoTokenizer.from_pretrained("/lustre/orion/csc569/scratch/jgeiping/tokenizers/huginn_tokenizer_65k")
    # Overwrite chat template to remove bos and eos. These are added later with standard tokenization
    tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|begin_header|>' + message['role'] + '<|end_header|>\n\n'+ message['content'] | trim + '<|end_turn|>' %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% endif %}"
    for name, source in data_sources.sources.items():
        print("__________________________________________________________________________________________")
        print(f"Source: {name}")
        print(f"  Original address: {source.address}")
        print(f"  New address: {DOWNLOADED_DATASETS_PATH}/{name}")
        print(f"  Features: {', '.join(source.features)}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}")
        print()

        args = {
            "streaming": False,
            "cache_dir": CACHE_DIR,
            "num_proc": num_proc,
            "download_config": datasets.DownloadConfig(cache_dir=CACHE_DIR, local_files_only=True),
            # "verification_mode": "NO_CHECKS", # does not work offline????
        }
        local_address = f"{DOWNLOADED_DATASETS_PATH}/{name}"
        datasets.utils._filelock.FileLock = contextlib.nullcontext
        try:
            if source.subset:
                args |= {"name": source.subset}
            if "dolma" in name or "redpajama" in name and "prox" not in name:
                dataset = datasets.load_dataset(
                    "parquet",
                    data_files=f"{DOWNLOADED_DATASETS_PATH}/{name}/*.parquet",
                    cache_dir=CACHE_DIR,
                    num_proc=num_proc,
                )
            elif "starcoder" in name and "smollm" not in name:
                dataset = datasets.load_dataset(
                    f"{DOWNLOADED_DATASETS_PATH}/starcoder",
                    data_dir=source.subset,
                    cache_dir=CACHE_DIR,
                    num_proc=num_proc,
                )
            elif "Txt360" in name:
                dataset = datasets.load_dataset(f"{DOWNLOADED_DATASETS_PATH}/TxT360", **args)
            elif "MathPile" in name:
                dataset = get_mathpile()
            elif name == "opus-writingprompts":
                dataset = datasets.load_dataset(local_address, data_files="*sharegpt.jsonl", **args)
            elif name == "reddit-writing":
                dataset = datasets.load_dataset(local_address, data_files="reddit_writing_prompts.jsonl", **args)
            elif "matrix-books" in name or "matrix-exam" in name:
                base_path = f"{DOWNLOADED_DATASETS_PATH}/matrix-books"
                dataset = load_and_concatenate_jsonl_files(name, source, source.subset, base_path)
            elif "proofpile" in name:
                data_files = glob.glob(f"{local_address}/{source.subset}/train/*.zst", recursive=True)
                dataset = load_text_only(data_files)
            elif "AutoMathText-1" in name:
                dataset = datasets.load_dataset(local_address, args.pop("name"), features=automath1_features, **args)
            elif "AutoMathText-2" in name:
                dataset = datasets.load_dataset(local_address, args.pop("name"), features=automath2_features, **args)
            elif "AutoMathText-3" in name:
                data_files = glob.glob(f"{local_address}/data/code/**/0.[5-9]*.jsonl", recursive=True)
                dataset = load_json_only(data_files)

            elif "together-long-data" in name:
                data_files = glob.glob(f"{local_address}/pretrain/*.zst", recursive=True)
                dataset = load_text_only(data_files)
            elif "the-stack-v2" in name:
                data_loc = "/lustre/orion/csc569/proj-shared/language_datasets/processed/bigcode/sean_proccessed/"
                folders = os.listdir(data_loc)
                chunks = []
                for folder in folders:
                    try:
                        ds = datasets.load_from_disk(f"{data_loc}/{folder}")
                        chunks += [ds]
                    except Exception as e:
                        print(f"{folder}: {e}")
                        raise
                dataset = datasets.concatenate_datasets(chunks)
            else:
                try:
                    dataset = datasets.load_dataset(local_address, **args)
                except ValueError:
                    if "name" in args:  # try loading with default + data_dir
                        args["data_dir"] = args.pop("name")
                        dataset = datasets.load_dataset(local_address, **args)
                    else:
                        dataset = load_dataset_without_default_split(local_address, args)
            if isinstance(dataset, datasets.DatasetDict):
                print(f"Concatenating splits {list(dataset.keys())}")  # some sources split topics as splits
                dataset = datasets.concatenate_datasets(list(dataset.values()))
            assert isinstance(dataset, datasets.Dataset)
            print(f"Source {name} ok.")

            # Already processed?

            if resume and check_if_shards_are_written(dataset, name):
                print(f"No unfinished shards left to write for dataset {name}")
                continue

            # Unify formatting for text + chat dataset
            remove_columns = [c for c in dataset.column_names if c != "text"]  # type: ignore
            if len(source.features) > 1 or source.features[0] != "text" or source.needs_chat_templating:
                formatter = get_unifying_formatter(source.features, tokenizer, source.needs_chat_templating)
                print("Running formatting unifier...")
                if "the-stack-v2" in name:
                    # need a special rule for load_from_disk datasets because the original file loc is not writiable
                    # and HF datasets does not pick up CACHE_DIR for load_from_disk ...
                    dataset = dataset.map(
                        formatter,
                        remove_columns=remove_columns,
                        num_proc=num_proc,
                        cache_file_name=f"{CACHE_DIR}/stack/format_cache.arrow",
                    )
                else:
                    dataset = dataset.map(formatter, remove_columns=remove_columns, num_proc=num_proc)  # type: ignore
            elif len(source.features) == 1 and source.features[0] != "text":
                dataset = dataset.rename_columns({source.features[0]: "text"})
                dataset = dataset.select_columns(["text"])
            else:
                dataset = dataset.select_columns(["text"])

            if filter_language:
                dataset = dataset.filter(language_filter, **args)  # Filter for English:
            if run_ftfy:
                dataset = dataset.map(format_ftfy, batch_size=512, **args)

            # Finally save into a unified parquet format.
            save_as_parquet(dataset, name, num_proc=0)  # messy with num_proc > 0, filesystem?
        except Exception as e:
            print(e)
            # raise


import dataclasses

from datatrove.pipeline.filters.sampler_filter import SamplerFilter
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.stats import StatsMerger, TopKConfig, TokenStats
from datatrove.executor import LocalPipelineExecutor

TOTAL_TASKS = 32

DATASET_STATS = f"{BASE_DIR}/stats_dataset"
LOCAL_LOGS_FOLDER = f"{DATASET_STATS}/trove-logs"


def datatrove_steps(experiment_name="stats_details"):
    top_k_config = TopKConfig(top_k_groups=["fqdn", "suffix"], top_k=10_000)
    data_sources = DataSources.from_yaml("scripts/sources2.yaml")
    merger = None
    for name, source in data_sources.sources.items():
        print("__________________________________________________________________________________________")
        print(f"Source: {name}")
        print(f"  Address: {source.address}")
        print(f"  New address: {DATASET_STAGING}/{name}")
        print(f"  Features: {', '.join(source.features)}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}")
        print()
        if Path(f"{DATASET_STAGING}/{name}").exists():
            tasks = min(
                sum(1 for file in os.listdir(f"{DATASET_STAGING}/{name}") if file.endswith(".parquet")), TOTAL_TASKS
            )
            print(f"Detected {tasks} tasks in this dataset")

            compute = LocalPipelineExecutor(
                pipeline=[
                    ParquetReader(
                        f"{DATASET_STAGING}/{name}", doc_progress=True, glob_pattern="**/*.parquet", text_key="text"
                    ),
                    SamplerFilter(rate=0.1),  # Sampling is fine for summary stats?
                    # WordStats(output_folder=f"{DATASET_STATS}/{name}", groups_to_compute=["histogram", "summary"]),
                    # LineStats(output_folder=f"{DATASET_STATS}/{name}", groups_to_compute=["histogram", "summary"]),
                    # DocStats(output_folder=f"{DATASET_STATS}/{name}", groups_to_compute=["histogram", "summary"]),
                    # ParagraphStats(output_folder=f"{DATASET_STATS}/{name}", groups_to_compute=["histogram", "summary"]),
                    TokenStats(
                        output_folder=f"{DATASET_STATS}/{name}",
                        tokenizer_name_or_path="/lustre/orion/csc569/scratch/jgeiping/tokenizers/huginn_tokenizer_65k/tokenizer.json",
                        groups_to_compute=["histogram", "summary"],
                    ),
                    # WordsContaminationStats(output_folder=f"{DATASET_STATS}/{name}", words=["As an AI language model"], top_k_config=top_k_config),
                ],
                tasks=TOTAL_TASKS,
                # job_name=f"summary-stats-{name}-{experiment_name}",
                # time="24:00:00",
                # partition="batch",
                logging_dir=f"{LOCAL_LOGS_FOLDER}-{name}-compute",
                # qos="normal",
                # mem_per_cpu_gb=2,
                # cpus_per_task=1,
                # sbatch_args={"account": "CSC569", "nodes" :  1},
                depends=merger,  # type: ignore
            )
            merger = LocalPipelineExecutor(
                pipeline=[
                    StatsMerger(
                        input_folder=f"{DATASET_STATS}/{name}",
                        output_folder=f"{DATASET_STATS}/{name}",
                        remove_input=False,
                        top_k_config=dataclasses.replace(top_k_config, top_k=8_000),
                    ),
                ],
                tasks=TOTAL_TASKS,
                # job_name=f"merging-stats-{name}-{experiment_name}",
                # time="24:00:00",
                # partition="batch",
                logging_dir=f"{LOCAL_LOGS_FOLDER}-{name}-merge",
                # qos="normal",
                # mem_per_cpu_gb=2,
                # cpus_per_task=1,
                # sbatch_args={"account": "CSC569", "nodes" :1},
                depends=compute,
            )

            merger.run()


if __name__ == "__main__":
    verify_dataset_can_be_loaded_and_process()
    # datatrove_steps()


import time
from huggingface_hub import snapshot_download


def download_dataset(
    repo_id: str,
    base_dir: str,
    dataset_folder: str,
    cache_dir: str,
    num_proc: int = 32,
    max_retries: int = 20,
    initial_retry_delay: int = 7,
):
    """
    Download a dataset from Hugging Face with robust error handling and retries

    Args:
        repo_id: Hugging Face repository ID
        base_dir: Base directory for downloads
        dataset_folder: Folder name for the dataset
        cache_dir: Cache directory path
        num_proc: Number of concurrent download processes
        max_retries: Maximum number of retry attempts
        initial_retry_delay: Initial delay between retries (will increase exponentially)
    """
    downloaded_path = os.path.join(base_dir, dataset_folder)

    # Create directories if they don't exist
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    for retry in range(max_retries):
        try:
            print(f"Attempting download of {repo_id} (Attempt {retry + 1}/{max_retries})")

            snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=downloaded_path,
                max_workers=num_proc,
                cache_dir=cache_dir,
                token=os.getenv("HF_TOKEN"),  # Optional: Use API token if available
            )

            print(f"Successfully downloaded {repo_id} to {downloaded_path}")
            return True

        except Exception as e:
            retry_delay = initial_retry_delay * (2**retry)  # Exponential backoff
            print(f"Error during download: {type(e).__name__}: {str(e)}")

            if retry < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retry {retry + 2}...")
                time.sleep(retry_delay)
            else:
                print(f"Failed to download after {max_retries} attempts")
                raise


def quick_downloader():
    # Configuration
    BASE_DIR = "/lustre/orion/csc569/scratch/jgeiping/data"
    DOWNLOADED_DATASETS_PATH = f"{BASE_DIR}/test_download_folder"
    CACHE_DIR = f"{BASE_DIR}/test_cache"
    REPO_ID = "LLM360/TxT360"

    try:
        download_dataset(
            repo_id=REPO_ID,
            base_dir=DOWNLOADED_DATASETS_PATH,
            dataset_folder="TxT360",
            cache_dir=CACHE_DIR,
            num_proc=32,
            max_retries=20,
            initial_retry_delay=7,
        )
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        raise
