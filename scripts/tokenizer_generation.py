"""This script generated the tokenizer."""

from pydantic import BaseModel
from typing import List, Optional, TYPE_CHECKING
import yaml
import os
from datetime import datetime
import subprocess

from pathlib import Path
import io
import contextlib
import boto3
import smart_open
import json
import math

from botocore import UNSIGNED
from botocore.config import Config

# current consts:
DOWNLOADED_DATASETS_PATH = "outputs/test_download_folder"
CACHE_DIR = "outputs/test_cache"
FINAL_LOCATION = "outputs/test_processed_dataset"
LIMITER = 2**17  # how many rows to take from each dataset in streaming mode
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = CACHE_DIR

import datasets
import huggingface_hub

# somehow required otherwise the token is not always read correctly:
huggingface_hub.login(token=os.environ.get("MANUALLY_SET_HF_TOKEN", None), add_to_git_credential=True)

if TYPE_CHECKING:
    import datasets.config


datasets.config.STREAMING_READ_MAX_RETRIES = 50
datasets.config.STREAMING_READ_RETRY_INTERVAL = 10

num_proc = 48
have_enough_ram = True
os.environ["RAYON_NUM_THREADS"] = str(num_proc)

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

# Fix broken security init in huggingface_hub

import huggingface_hub


def patched_repofolder_init(self, **kwargs):
    self.path = kwargs.pop("path")
    self.tree_id = kwargs.pop("oid")
    last_commit = kwargs.pop("lastCommit", None) or kwargs.pop("last_commit", None)
    if last_commit is not None:
        last_commit = huggingface_hub.hf_api.LastCommitInfo(
            oid=last_commit["id"],
            title=last_commit["title"],
            date=huggingface_hub.utils.parse_datetime(last_commit["date"]),
        )
    self.last_commit = last_commit


def patched_repo_file_init(self, **kwargs):
    self.path = kwargs.pop("path")
    self.size = kwargs.pop("size")
    self.blob_id = kwargs.pop("oid")
    lfs = kwargs.pop("lfs", None)
    if lfs is not None:
        lfs = huggingface_hub.hf_api.BlobLfsInfo(size=lfs["size"], sha256=lfs["oid"], pointer_size=lfs["pointerSize"])
    self.lfs = lfs
    last_commit = kwargs.pop("lastCommit", None) or kwargs.pop("last_commit", None)
    if last_commit is not None:
        last_commit = huggingface_hub.hf_api.LastCommitInfo(
            oid=last_commit["id"],
            title=last_commit["title"],
            date=huggingface_hub.utils.parse_datetime(last_commit["date"]),
        )
    self.last_commit = last_commit
    self.security = None

    # backwards compatibility
    self.rfilename = self.path
    self.lastCommit = self.last_commit


huggingface_hub.hf_api.RepoFile.__init__ = patched_repo_file_init
huggingface_hub.hf_api.RepoFolder.__init__ = patched_repofolder_init

datasets.disable_caching()
datasets.config.DOWNLOADED_DATASETS_PATH = Path(DOWNLOADED_DATASETS_PATH)


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


def stream_to_local(streaming_dataset):
    return datasets.Dataset.from_generator(streaming_dataset.__iter__)


def software_heritage_aws_download(dataset):
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    def download_contents(files):
        contents = []
        for repo_file in files:
            s3_url = f"s3://softwareheritage/content/{repo_file['blob_id']}"

            with smart_open.open(s3_url, "rb", compression=".gz", transport_params={"client": s3}) as fin:
                # contents.append({"file": repo_file["path"], "text": fin.read().decode(repo_file["src_encoding"])})
                contents.append(fin.read().decode(repo_file["src_encoding"]))

        return {"content": "\n".join(contents)}

    dataset = dataset.map(lambda row: download_contents(row["files"]))
    return dataset


def load_dataset_without_default_split(source, args):
    print("Attempting to fix by detecting and concatenating all subsets.")
    configs = datasets.get_dataset_config_names(source.address)
    splits = []
    for config in configs:
        print(f"Loading {config}")
        subset = datasets.load_dataset(source.address, config, **args)
        if isinstance(subset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
            subset = subset[list(subset.keys())[0]]
        splits.append(subset)
    dataset = datasets.concatenate_datasets(splits)
    return dataset


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

    # Create a streaming dataset directly from local JSONL files
    dataset = datasets.IterableDataset.from_generator(generator_from_json(jsonl_files))
    return dataset


def save_shard(dataset, output_path_template, index, num_shards):
    output_path = output_path_template.format(index=index)
    if not Path(output_path).exists():
        shard = dataset.shard(index=index, num_shards=num_shards, contiguous=True)
        shard.to_parquet(output_path)
        print(f"Saved shard {index + 1}/{num_shards}")


def save_as_parquet(dataset, name, target_shard_size_bytes=2 * 1024 * 1024 * 1024, num_proc=32):
    # Calculate number of shards
    num_shards = math.ceil(dataset.info.size_in_bytes / target_shard_size_bytes)

    print(f"Dataset size according to HF: {dataset.info.size_in_bytes / (1024**3):.2f} GB")
    print(f"Target shard size: {target_shard_size_bytes / (1024**3):.2f} GB")
    print(f"Number of shards: {num_shards}")

    output_path_template = os.path.join(FINAL_LOCATION, name, "{index:05d}.parquet")
    unfinished_shards = [
        index for index in range(num_shards) if not Path(output_path_template.format(index=index)).exists()
    ]
    # Create a pool of worker processes
    #  with mp.Pool(processes=num_proc) as pool:
    #     pool.map(save_shard_partial, unfinished_shards)
    for shard in unfinished_shards:
        save_shard(dataset, output_path_template, shard, num_shards=num_shards)


names_for_roles = ["role", "from"]
names_for_content = ["content", "value"]


def unify_formatting(dataset, name, features, needs_chat_templating=False):
    def process_example(example):
        if name == "ChatQA-sft":
            output = example["messages"] + [{"role": "assistant", "content": str(example["answers"])}]
        elif len(features) > 1:
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
            example["text"] = ""
            for message in output:
                if message["content"] is not None:  # type: ignore
                    example["text"] = example["text"] + "\n" + message["content"]  # type: ignore
        else:
            # example["messages"] = []
            example["text"] = output
        return example

    # Special handling for terrible interactions between datasets
    # (not parsing columns correctly) and badly formatted datasets (having terrible columns)
    if name == "OpenMathInstruct":
        terrible_column_names = [
            "expected_answer",
            "predicted_answer",
            "error_message",
            "question",
            "generated_solution",
            "is_correct",
            "generation_type",
            "dataset",
        ]
    elif name == "TemplateGSM":
        terrible_column_names = [
            "problem",
            "solution_code",
            "result",
            "solution_wocode",
            "source",
            "template_id",
            "problem_id",
        ]
    elif name == "openweb-math":
        terrible_column_names = ["url", "date", "metadata", "openwebmath_perplexity", "openwebmath_score"]
    elif "AutoMathText" in name:
        terrible_column_names = ["meta"]
    else:
        terrible_column_names = None
    # Combine all processing steps into a single map operation
    remove_columns = (
        [c for c in dataset.column_names if c not in ["messages", "text"]]
        if dataset.column_names
        else terrible_column_names
    )
    try:
        result = dataset.map(lambda row: process_example(row), remove_columns=remove_columns)
    except Exception as e:
        print(e)
        result = dataset.map(lambda row: process_example(row))
    return result


def create_dataset(shuffle_streaming_shards=True):
    # data_sources = DataSources.from_yaml("scripts/sources.yaml")
    data_sources = DataSources.from_yaml("scripts/target_domain_sets.yaml")

    data_handles = {}
    for name, source in data_sources.sources.items():
        print("__________________________________________________________________________________________")
        print(f"Source: {name}")
        print(f"  Address: {source.address}")
        print(f"  Features: {', '.join(source.features)}")
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S.%f}")
        print()
        try:
            args = {"streaming": True, "cache_dir": CACHE_DIR, "trust_remote_code": True, "token": True}
            if source.subset:
                args |= {"name": source.subset}
            if "dolma" in name:
                dataset = datasets.load_dataset("recpre/data/dolma.py", **args)
            elif "MathPile" in name:  # special rules for mathpile...
                dataset = custom_mathpile_download()
            elif name == "opus-writingprompts":
                dataset = datasets.load_dataset(source.address, data_files="*sharegpt.jsonl", **args)
            elif name == "reddit-writing":
                dataset = datasets.load_dataset(source.address, data_files="reddit_writing_prompts.jsonl", **args)
            else:
                try:
                    dataset = datasets.load_dataset(source.address, **args)
                except ValueError:
                    if "name" in args:  # try loading with default + data_dir
                        args["data_dir"] = args.pop("name")
                        dataset = datasets.load_dataset(source.address, **args)
                    else:
                        dataset = load_dataset_without_default_split(source, args)
            if isinstance(dataset, (datasets.DatasetDict, datasets.IterableDatasetDict)):
                print(f"Selecting split {list(dataset.keys())[0]}")
                dataset = dataset[list(dataset.keys())[0]]

            if source.requires_software_heritage_aws_download:
                dataset = software_heritage_aws_download(dataset)

            # Combine all processing steps into a single map operation
            dataset = unify_formatting(dataset, name, source.features, source.needs_chat_templating)
            print("----------------Pre-formatted: -------------------")
            print(next(iter(dataset))["text"][:500])

            # Shuffle and reduce rows
            if shuffle_streaming_shards:
                dataset = dataset.shuffle(buffer_size=2**14, seed=233)  # type: ignore[thanks static typing]
            if LIMITER is not None:
                dataset = dataset.take(int(LIMITER))

            # Prepare for stream to disk
            dataset = datasets.Dataset.from_generator(
                dataset.__iter__,
                # num_proc=num_proc,  # need to write better code to use this
                keep_in_memory=have_enough_ram,
                cache_dir=CACHE_DIR,
            )
            # 2nd attempt to remove columns
            dataset = dataset.select_columns(["text"])  # type: ignore

            assert name not in data_handles
            data_handles[name] = dataset
        except Exception as e:
            print(e)
            # raise
    monolith = datasets.DatasetDict(data_handles)

    monolith.save_to_disk(FINAL_LOCATION, max_shard_size="4096MB", num_proc=num_proc)
    print("Dataset processing and export successful")


def get_regex_from_normalization_rule_name(normalization_rule_name: str) -> str:
    # GPT4 regex
    if normalization_rule_name == "gpt":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    elif normalization_rule_name == "gpt-2":
        return r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k to ensure full digit coverage)
    elif normalization_rule_name == "gpt-num2":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # separates punctuation from words (except spaces)
    elif normalization_rule_name == "punct":
        return r""" ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k to ensure full digit coverage)
    elif normalization_rule_name == "punct-num2":
        return r""" ?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    elif normalization_rule_name == "gpt-num1":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    elif normalization_rule_name == "slop":
        return r"(\s*)(?:(?<!\w)(?:[a-zA-Z]+[''](?:d|ll|m|re|s|t|ve)\b)|\$\$(?:[^$]|\$(?!\$))*\$\$|\$(?:[^$]|\$(?!\$))*\$|\\begin\{[^}]+\}(?:(?!\\end\{).)*?\\end\{[^}]+\}|\\[a-zA-Z]+(?:\[[^]]*\])?(?:\{[^}]*\})+|[a-zA-Z0-9]+(?:[_^](?:\{[a-zA-Z0-9,]+\}|[a-zA-Z0-9]))+|```[\s\S]*?```|`[^`\n]+`|https?://\S+|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?|-?(?:\d*\.)?\d+(?:[eE][-+]?\d+)?(?:[a-zA-Z]+(?:[-^/][a-zA-Z0-9]+)*)?|[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*|</?[a-zA-Z][^>]*>|==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|[a-zA-Z]+(?:[''][a-z]+)?|[!-/:-@\[-`{-~]|\s+)"
    elif normalization_rule_name == "hell":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n|\\n(?!\b)|\s*(?:[\p{L}]+[''](?:\p{L}+)\b|\$\$?|\\[a-zA-Z]+|`+|\"+|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?|\p{N}+(?:st|nd|rd|th)|\p{N}{1,1}|[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|<[^>]*>|</[^>]*>|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}]+)|\s+|\p{So})"
    elif normalization_rule_name == "of-our-own-making":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n|\\n(?!\b)|[ \t]?(?:==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+[''](?:\p{L}+)\b|\\{|\\}|{|}|\[|\]|\(|\)|_+|\$\$?|\\[a-zA-Z]+|`+|\"+|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?|\p{N}+(?:st|nd|rd|th)|\p{N}{1,1}|[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*|<[^>]*>|</[^>]*>|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}])|\s+|\p{So})"
    elif normalization_rule_name == "hel":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n|\\n(?!\b)|[ \t]?(?:==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+['’ʼ](?:\p{L}+)\b|\\{|\\}|{|}|\[|\]|\(|\)|_+|\||\+|-|\*|%|—|[.,!?;()]|\$\$?|\\[a-zA-Z]+|`+|\"+|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}(?::\d{2})?(?:\s*[ap]\.?m\.?)?|\p{N}+(?:st|nd|rd|th)|\p{N}{1,1}|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}])|\s+|\p{So})"
    elif normalization_rule_name == "hesiod":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n+|\\n(?!\b)|[ \t]?(?:==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+['’ʼ](?:\p{L}+)\b|\\[a-zA-Z]+|\\{|\\}|{|}|\\\[|\\\]|\[|\]|\(|\)|_+|\||\+|-|\*|—|-|=|[.,!?;()]|\$\$?|`+|\"+|\#+|\p{N}+(?:st|nd|rd|th)|\p{N}{1,1}|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}]+)|\s+|\p{So})"
    elif normalization_rule_name == "cyme":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n+|\\n(?!\b)|\p{N}{1,1}|[ \t]?(?: ==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+['’ʼ](?:\p{L}+)\b|\\[a-zA-Z]+|\\{|\\}|{|}|\\\[|\\\]|\[|\]|\(|\)|_+|\||\+|-|\*|—|-|=|[.,!?;()]|\$\$?|`+|\"+|\#+|\p{N}+(?:st|nd|rd|th)|<0x[0-9A-Fa-f]{1,2}>|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}])|\s+|\p{So})"
    elif normalization_rule_name == "aeolis":
        return ""  # borked r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n+|\\n(?!\b)|\p{N}{1,1}|[ \t]?(?: ==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+['’ʼ](?:\p{L}+)\b|\\[a-zA-Z]+|\\{|\\}|{|}|\\\[|\\\]|\[|\]|\(|\)|_+|\||\+|-|\*|—|-|=|\*\*|^|[.,!?;()]|\$\$?|`+|\"+|\#+|\p{N}+(?:st|nd|rd|th)|<0x[0-9A-Fa-f]{1,2}>|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}]+)|\s+|\p{So})"
    elif normalization_rule_name == "cymean":
        return ""  # borked r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n+|\\n(?!\b)|\p{N}{1,1}|[ \t]?(?: ==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+['’ʼ](?:\p{L}+)\b|\\[a-zA-Z]+|\\{|\\}|{|}|\\\[|\\\]|\[|\]|\(|\)|_+|\||\+|-|\*|—|-|=|\*\*|^|[.,!?;()]|\$\$?|`+|\"+|\#+|\p{N}+(?:st|nd|rd|th)|<0x[0-9A-Fa-f]{1,2}>|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}])|\s+|\p{So})"
    elif normalization_rule_name == "ascra":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n+|\\n(?!\b)|\p{N}{1,1}|(?:[ \t](?:==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+[''ʼ](?:\p{L}+)\b|\\[a-zA-Z]+|\\{|\\}|{|}|\\\[|\\\]|\[|\]|\(|\)|_+|\||\+|-|\*|—|-|=|\*\*|^|[.,!?;()]|\$\$?|`+|\"+|\#+|\p{N}+(?:st|nd|rd|th)|<0x[0-9A-Fa-f]{1,2}>)|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}]+)|\s+|\p{So})"
    elif normalization_rule_name == "aeolian":
        return r"(\\n(?:e(?:wcommand|wenvironment|wline|wpage|wtheorem)|abla|e|eq|geq|gtr|leq|exists|ocite|oindent|olinebreak|onumber|opagebreak|ormalsize|ot(?!(?:ag\b|\b)))\b|\n+|\\n(?!\b)|\p{N}{1,1}|(?:[ \t](?:==|!=|<=|>=|<<|>>|\*\*|//|\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<=|>>=|\*\*=|//=|:=|\+\+|--|[\p{L}]+[''ʼ](?:\p{L}+)\b|\\[a-zA-Z]+|\\{|\\}|{|}|\\\[|\\\]|\[|\]|\(|\)|_+|\||\+|-|\*|—|-|=|\*\*|^|[.,!?;()]|\$\$?|`+|\"+|\#+|\p{N}+(?:st|nd|rd|th)|<0x[0-9A-Fa-f]{1,2}>)|[^\r\n\p{L}\p{N}]?\p{L}+|[\p{P}\p{S}])|\s+|\p{So})"
    else:
        raise ValueError(f"Unknown normalization_rule_name {normalization_rule_name}")


# Regex Pattern Differences:
#
# | Feature               | Hell          | Of-Our-Own    | Hesiod        | Cyme          | Ascra         | Aeolian       |
# |-----------------------|---------------|---------------|---------------|---------------|---------------|---------------|
# | Punctuation grouping  | [\p{P}\p{S}]+ | [\p{P}\p{S}]+ | [\p{P}\p{S}]+ | [\p{P}\p{S}]  | [\p{P}\p{S}]+ | [\p{P}\p{S}]  |
# | Whitespace before ops | \s* (0+)      | [ \t]? (0-1)  | [ \t]? (0-1)  | [ \t]? +extra | [ \t]? (0-1)  | [ \t] (1)     |
# | Newlines              | \n            | \n            | \n+           | \n+           | \n+           | \n+           |
# | Tags/Hex              | <[^>]*>       | <[^>]*>       | <[^>]*>       | hex codes     | hex codes     | hex codes     |
# | Single digit handling | With general  | With general  | With general  | Isolated      | Isolated      | Isolated      |
#
# Notes:
# - Punctuation grouping affects how symbols are tokenized:
#   - [\p{P}\p{S}]+ groups consecutive punctuation as one token
#   - [\p{P}\p{S}] treats each punctuation mark separately
# - Whitespace handling varies significantly:
#   - Hell uses \s* (any amount of any whitespace)
#   - Most patterns use [ \t]? (optional single space/tab)
#   - Cyme has an extra space in operator group
#   - Aeolian requires exactly one space/tab
# - Single digit handling is a key differentiator:
#   - Earlier patterns handle single digits in general number matching
#   - Cyme onwards isolates bare single digits (\p{N}{1,1}) in separate pattern
#   - This means "1" matches differently from " 1" in later patterns
# - Later patterns replace HTML/XML tag matching with hex code support


def get_special_tokens(layout="base"):
    if layout == "base":
        return ["<s>", "<pad>", "</s>"]
    elif layout == "complete":
        return [
            "<|begin_text|>",
            "<|end_text|>",
            "<|begin_header|>",
            "<|end_header|>",
            "<|end_turn|>",
            "<|pad|>",
            # unused, but included for completeness
            "<|fim_prefix|>",
            "<|fim_middle|>",
            "<|fim_suffix|>",
            "<|fim_pad|>",
            "<|...|>",
            "<|del|>",
            "<|c_0|>",
            "<|c_1|>",
            "<|c_2|>",
            "<|c_3|>",
            "<|c_4|>",
            "<|c_5|>",
            "<|c_6|>",
            "<|c_7|>",
            "<|c_8|>",
            "<|c_9|>",
            #
            "<|?|>",
            "<|&|>",
            "<|!|>",
            "<|-|>",
            "<|+|>",
            "<|interrupt|>",
            "<|>>-|>",
            "<|-<<|>",
            "<|->>|>",
            "<|<<-|>",
            "<|tool_out|>",
            "<|end_tool|>",
            "<|tool_in|>",
            "<|end_tool_in|>",
        ]
    elif layout == "starcoder":
        return [
            "<|endoftext|>",
            "<fim_prefix>",
            "<fim_middle>",
            "<fim_suffix>",
            "<fim_pad>",
            "<filename>",
            "<gh_stars>",
            "<issue_start>",
            "<issue_comment>",
            "<issue_closed>",
            "<jupyter_start>",
            "<jupyter_text>",
            "<jupyter_code>",
            "<jupyter_output>",
            "<empty_output>",
            "<commit_before>",
            "<commit_msg>",
            "<commit_after>",
            "<reponame>",
        ]
    elif layout == "llama3":
        return [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",  # end of turn
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    else:
        raise ValueError


emoji_ranges = [
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F300, 0x1F5FF),  # Miscellaneous Symbols and Pictographs
    (0x1F680, 0x1F6FF),  # Transport and Map Symbols
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x2600, 0x26FF),  # Miscellaneous Symbols
    (0x2700, 0x27BF),  # Dingbats
    # (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    # (0x1FA00, 0x1FA6F),  # Supplemental Symbols and Pictographs Extended
    # (0x1FB00, 0x1FBFF),  # Symbols for Legacy Computing
    # (0x1FBC0, 0x1FBFF),  # Symbols and Pictographs Extended-B
    # (0x1F100, 0x1F1FF),  # Enclosed Alphanumeric Supplement
    # (0x2460, 0x24FF),  # Enclosed Alphanumerics
]

math_ranges = [
    (0x2150, 0x218F),  # Number Forms
    (0x2190, 0x21FF),  # Arrows
    (0x2200, 0x22FF),  # Mathematical Operators
]


def get_manually_added_tokens():
    return (
        ['"""', "'''", '"', "''", 'f"', "f'"]
        + [" " * x for x in range(1, 17)]  # coding
        + ["\n" * x for x in range(1, 17)]  # paragraphs
        + ["\t" * x for x in range(1, 17)]
        + [chr(i) for i in range(687)]  # guarantee basic unicode coverage - latin+IPA
        + [chr(129453)]  # critical
        + [chr(i) for start, end in emoji_ranges for i in range(start, end + 1)]  # do I want these?
        + [chr(i) for start, end in math_ranges for i in range(start, end + 1)]
        + ["Huginn"]
    )


def create_iterator_without_roles(dataset):
    for example in iter(dataset):
        if "messages" in example and len(example["messages"]) > 0:
            text = ""
            for message in example["messages"]:
                text = text + "\n" + message["content"]
        else:
            text = example["text"]
        yield text[: 8 * 4096]  # limit max chars per row to reduce self-similarity


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
        # print("------------------------------------------------------------------------------------------------------")
        # print(f"Filtered: {lang} ({lang_score:.2f})")
        # print(doc.text)
        return False
    return True


def create_test_parquet():
    dataset: datasets.DatasetDict = datasets.load_dataset(
        FINAL_LOCATION, cache_dir=CACHE_DIR, keep_in_memory=have_enough_ram
    )  # type: ignore

    monolith = datasets.concatenate_datasets(list(dataset.values()))
    # Finally save into a unified parquet format.
    save_as_parquet(monolith, "test_v1", num_proc=1)


def create_tokenizer(vocab_size=65536, regex_rule="aeolian"):
    dataset: datasets.DatasetDict = datasets.load_dataset(
        FINAL_LOCATION, cache_dir=CACHE_DIR, keep_in_memory=have_enough_ram
    )  # type: ignore

    monolith = datasets.concatenate_datasets(list(dataset.values()))
    regex_pattern = get_regex_from_normalization_rule_name(regex_rule)
    special_tokens = get_special_tokens("complete")
    manual_tokens = get_manually_added_tokens()

    # Filter for English:
    monolith = monolith.filter(language_filter, keep_in_memory=have_enough_ram, num_proc=num_proc)
    print("English filtering done, creating iterator now")
    iterator = create_iterator_without_roles(monolith)

    from bpeasy.tokenizer import train_bpe, BPEasyTokenizer

    # Use BPE to compute vocab
    print("Data loaded, starting tokenizer generation")
    bytes_vocab = train_bpe(
        iterator,
        python_regex=regex_pattern,
        max_token_length=32,
        vocab_size=vocab_size - len(special_tokens) - len(manual_tokens),
    )
    print("BPE phase survived")
    if len(bytes_vocab) < vocab_size - len(special_tokens) - len(manual_tokens):
        raise ValueError("Insufficient data for full vocabulary construction.")
    # Add manual tokens
    reserved_slots = 0
    for token in manual_tokens:
        correctly_encoded_token = token.encode("utf-8")
        if correctly_encoded_token not in bytes_vocab:
            bytes_vocab[correctly_encoded_token] = len(bytes_vocab)
        else:
            bytes_vocab[f"<|reserved_token_{reserved_slots}|>".encode("utf-8")] = len(bytes_vocab)
            reserved_slots += 1

    assert len(bytes_vocab) == vocab_size - len(special_tokens)

    tokenizer = BPEasyTokenizer(
        name="baseline_tokenize",
        vocab=bytes_vocab,
        regex_pattern=regex_pattern,
        special_tokens=special_tokens,
        fill_to_nearest_multiple_of_eight=True,
    )

    tokenizer.save(os.path.join(FINAL_LOCATION, f"bp_basis_{regex_rule}.json"))
    tokenizer.export_to_huggingface_format(os.path.join(FINAL_LOCATION, f"hf_tokenizer_{regex_rule}.json"))
    print("Tokenizer exported successfully.")


if __name__ == "__main__":
    # create_dataset()
    # create_test_parquet()
    create_tokenizer()
