# Build based on the original foundation from Lightning AI
# litgpt/packed_dataset.py - but completely with a new Parquet Dataset implementation


# Original Code:
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

# Very loosely inspired by indexed_dataset in Fairseq, Megatron
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/data/indexed_dataset.py


import random
import hashlib
from torch.utils.data import IterableDataset, get_worker_info
from pathlib import Path

# We will build v0 assuming that the dataset is already saved to disk
# in standard hf format. This leaves room for preproc ops as separate logic.
# basic assumpution will be "text" field only.
from datasets import load_from_disk, DatasetDict, concatenate_datasets, Dataset, load_dataset
from datasets import IterableDataset as HFIterableDataset

import logging
import glob

import torch
import pyarrow.parquet as pq


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ParquetStream(IterableDataset):
    def __init__(
        self,
        dataset_folder_path="",
        seed=12345,
        shuffle=True,
        num_processes=1,
        process_rank=0,
        data_id=None,
        data_signature: dict[str, list[str] | str] = {"keys": ["input_ids"]},
        repetitions=None,
        return_data_id=False,
        broadcast_glob=True,
        shuffle_filenames=True,
        prefix="",
        verbose=True,
        stateful=True,
    ):
        # Dist info:
        self._num_processes = num_processes
        self._process_rank = process_rank

        if broadcast_glob and torch.distributed.is_initialized():
            if process_rank == 0:
                filenames = [str(pth) for pth in sorted(Path(dataset_folder_path).glob(f"{prefix}*"))]
                if shuffle_filenames:
                    random.seed(seed)
                    random.shuffle(filenames)  # inplace
                if not filenames:
                    raise FileNotFoundError(f"No files found at {str(dataset_folder_path)} with prefix {prefix}.")
            else:
                filenames: list[str] = None  # type: ignore # hashtag believe
            obj = [filenames]
            torch.distributed.broadcast_object_list(obj, 0)  # this is a blocking op from rank 0 to all other ranks
            parquet_files = obj[0]
            # log after broadcast so we know we passed it.
            if process_rank == 0 and verbose:
                logger.info(
                    f"Rank ({process_rank}/{num_processes}) received {len(parquet_files)} files"
                    f" from {dataset_folder_path}{f' w/ prefix {prefix}' if prefix not in ['', '*'] else ''},"
                    f" files[:3]: {parquet_files[:3]}"
                )
        else:
            parquet_files = glob.glob(f"{dataset_folder_path}/**.parquet", recursive=True)
        if self._num_processes > 1:
            parquet_files = parquet_files[process_rank :: self._num_processes]
        self._ds_fingerprint = hashlib.shake_128(str(parquet_files).encode()).hexdigest(4)
        self._ds: HFIterableDataset = load_dataset("parquet", data_files=parquet_files, streaming=True, split="train")  # type: ignore
        self._ds = self._ds.with_format("torch")
        if shuffle:
            self._ds = self._ds.shuffle(buffer_size=2**10, seed=seed)  # type: ignore

        if verbose:
            logger.info(
                f"Rank {self._process_rank}/{self._num_processes} has "
                f"{len(parquet_files)} parquet files | identifier={data_id}:{self._ds_fingerprint}"
            )
            logger.info(
                f"Rank {self._process_rank}/{self._num_processes}. DATA: ---------------\n  {next(iter(self._ds))}"
            )
        self.parquet_files = parquet_files

    def __iter__(self):
        for entry in iter(self._ds):
            yield entry["input_ids"]

    def state_dict(self):
        return self._ds.state_dict()

    def load_state_dict(self, state_dict):
        self._ds.load_state_dict(state_dict)

    def iter_as_torch_loader(self, batch_size=128):
        """Sanity check loader, circumventing both HF, batching via torch.utils and threading in pyarrow"""
        for file in self.parquet_files:
            loaded_file = pq.ParquetFile(file)
            for record in loaded_file.iter_batches(batch_size=batch_size, use_threads=False):
                tokens = torch.as_tensor(record.to_pydict()["input_ids"], dtype=torch.long)
                yield tokens

    def iter_as_unbatched_parquet_loader(self):
        """Sanity check loader, circumventing HF"""
        for file in self.parquet_files:
            loaded_file = pq.ParquetFile(file)
            table = loaded_file.read()["input_ids"].to_numpy()
            for row in table:
                tokens = torch.tensor(row)
                yield tokens


class ParquetStreamPure(IterableDataset):
    """datasets-free version of (mostly) the same thing - shuffle not across files though
    a bit ironic to keep it in this file"""

    def __init__(
        self,
        dataset_folder_path="",
        seed=12345,
        shuffle=True,
        num_processes=1,
        process_rank=0,
        prefix="",
        verbose=False,
        shuffle_filenames=True,
        data_signature: dict[str, list[str] | str] = {"keys": ["input_ids"]},
        repetitions=None,
        return_data_id=False,
        data_id=None,
        broadcast_glob=True,
        stateful=True,  # this dataset is by default (suprisingly) stateful, even outside of iter(dataset)!
        plan_for_later_rank_expansion_to=8,
    ):
        # Get file list, with distributed broadcast if needed
        if broadcast_glob and torch.distributed.is_initialized():
            if process_rank == 0:
                filenames = sorted(str(p) for p in Path(dataset_folder_path).glob(f"{prefix}*.parquet"))
            else:
                filenames: list[str] = None  # type: ignore # believe
            obj = [filenames]
            torch.distributed.broadcast_object_list(obj, 0)
            parquet_files = obj[0]
        else:
            parquet_files = sorted(str(p) for p in Path(dataset_folder_path).glob(f"{prefix}*.parquet"))
        if shuffle_filenames:
            random.Random(seed).shuffle(parquet_files)

        # Shard files for distributed training
        if plan_for_later_rank_expansion_to > 0:
            ranks_for_file_selection = max(plan_for_later_rank_expansion_to, num_processes)
        else:
            ranks_for_file_selection = num_processes
        self.parquet_files = (
            parquet_files[process_rank::ranks_for_file_selection] if ranks_for_file_selection > 1 else parquet_files
        )
        if len(self.parquet_files) < 1:
            raise ValueError(f"Empty dataset on rank {self.process_rank}")
        self._ds_fingerprint = hashlib.shake_128(str(self.parquet_files).encode()).hexdigest(4)

        if verbose:
            logger.info(
                f"Rank {process_rank}/{num_processes} has {len(self.parquet_files)} parquet files | identifier={self._ds_fingerprint}"
            )
            examples = pq.read_table(self.parquet_files[0], columns=["input_ids"]).slice(0, 3).to_pylist()  # Get 3 rows
            for i, example in enumerate(examples):
                logger.info(f"Example {i}: {example['input_ids'][:12]}")  # First 12 tokens of each row
        self.shuffle = shuffle
        self.seed = seed
        self.process_rank = process_rank
        self.stateful = stateful
        # Initialize default state
        self._state_init()

    def _state_init(self):
        self._state = {
            "rng": random.Random(self.seed),
            "rng_state": (-1, [-1], None),
            "buffer": [],
            "file_idx": 0,
            "row_group_idx": 0,
            "fingerprint": self._ds_fingerprint,
        }

    def __iter__(self):
        if not self.stateful:
            self._state_init()

        while self._state["file_idx"] < len(self.parquet_files):
            if not self._state["buffer"]:
                # Refill buffer from current position
                pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
                if self._state["row_group_idx"] >= pf.num_row_groups:
                    print(
                        f"Rank {self.process_rank} | {self._state['file_idx']}-{self._state['row_group_idx']} | "
                        f" New file: {self.parquet_files[self._state['file_idx'] + 1]}"
                    )
                    self._state["file_idx"] += 1
                    self._state["row_group_idx"] = 0
                    continue

                self._read_buffer(pf)
                self._state["row_group_idx"] += 1

            while self._state["buffer"]:
                yield torch.as_tensor(self._state["buffer"].pop(), dtype=torch.long)

    def _read_buffer(self, parquet_file):
        batch = parquet_file.read_row_group(self._state["row_group_idx"])
        self._state["buffer"] = batch.column("input_ids").to_pylist()

        if self.shuffle:
            self._state["rng_state"] = self._state["rng"].getstate()  # the last used state for a shuffle op
            self._state["rng"].shuffle(self._state["buffer"])

    def state_dict(self):
        # Pack all basic state and RNG state into one tensor for a single gather
        rng_0, rng_1, rng_2 = self._state["rng_state"]
        local_state = torch.tensor(
            [
                self._state["file_idx"],
                self._state["row_group_idx"] - 1,  # -1 because we need to reload the current buffer
                len(self._state["buffer"]),
                int(self._ds_fingerprint, 16),
                rng_0,
                *rng_1,
                rng_2 if rng_2 is not None else -1,
            ],
            device="cuda",
        )

        # Single gather for all state
        gathered_states = [torch.zeros_like(local_state) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_states, local_state)

        result = {
            "file_idx": [s[0].item() for s in gathered_states],
            "row_group_idx": [s[1].item() for s in gathered_states],
            "row_idx": [s[2].item() for s in gathered_states],
            "fingerprint": [hex(s[3].item())[2:] for s in gathered_states],  # type: ignore
            "rng_state": gathered_states,  # Full tensors for unpacking in load
        }

        return result

    def load_state_dict(self, state_dict, offset_ranks=False):
        rank = torch.distributed.get_rank()

        def get_value(key):  # helper for backward compat
            effective_rank = rank % len(state_dict["fingerprint"])
            return state_dict[key][effective_rank]

        if int(get_value("fingerprint"), 16) != int(self._ds_fingerprint, 16):
            print(
                f"WARNING Dataset fingerprint mismatch. Expected {self._ds_fingerprint}, got {get_value('fingerprint')}"
            )
            self._state["file_idx"] = 0
            self._state["row_group_idx"] = 0
            row_idx = 0
        else:
            # Load file IDs only if we can guarantee that this the same data source
            # otherwise these might run out of bounds
            self._state["file_idx"] = max(get_value("file_idx"), 0)
            self._state["row_group_idx"] = max(get_value("row_group_idx"), 0)
            row_idx = max(get_value("row_idx"), 0)
            if offset_ranks:
                row_idx = row_idx + rank % 1000

        if state_dict["rng_state"] is not None:
            # New packed format
            rng_state = state_dict["rng_state"][rank % len(state_dict["rng_state"])]
            # RNG state starts at index 4
            rng_state = (rng_state[4].item(), tuple(x.item() for x in rng_state[5:-1]), rng_state[-1].item())
            if rng_state[2] == -1:
                rng_state = (rng_state[0], rng_state[1], None)

            self._state["rng"] = random.Random()
            self._state["rng"].setstate(rng_state)
            self._state["rng_state"] = rng_state

        pf = pq.ParquetFile(self.parquet_files[self._state["file_idx"]])
        self._read_buffer(pf)
        self._state["buffer"] = self._state["buffer"][:row_idx]


class HuggingfaceDataset(IterableDataset):
    def __init__(
        self,
        ds_name_or_path=None,
        seed=12345,
        shuffle=False,
        num_processes=1,
        process_rank=0,
        data_id=None,
        data_signature: dict[str, list[str] | str] = {"keys": ["text"], "format_fn": "pass_text"},
        repetitions=None,
        return_data_id=False,
    ):
        assert ds_name_or_path is not None
        self._ds_name_or_path = ds_name_or_path
        self._seed = seed
        assert not shuffle, "Shuffle not implemented for hfds."
        self._num_processes = num_processes
        self._process_rank = process_rank
        self._data_id = data_id  # This is human readble, the mixture unit
        self._return_data_id = return_data_id
        self._ds_fingerprint = (
            None  # This is not human readable, corresp to the subset of work _this_ process is handling.
        )
        self._data_signature = data_signature
        self._ds_total_length = None
        self._ds_length = None
        self._subds = None  # type: ignore
        self._ds_min = None
        self._ds_max = None

        # Here is where we load the dataset from disk (whole thing, but just the memmap ofc)
        if repetitions is not None:
            ds_list = [load_from_disk(ds_name_or_path) for _ in range(repetitions)]
            self._ds: Dataset = concatenate_datasets(ds_list)  # type: ignore
        else:
            self._ds: Dataset = load_from_disk(ds_name_or_path)  # type: ignore

        assert not isinstance(self._ds, DatasetDict), (
            "Dataset path should point to a single split, try adding /train ?."
        )

        self._ds_total_length = len(self._ds)

    def __iter__(self):  # type: ignore
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        num_shards = num_workers * self._num_processes
        shard_id = self._process_rank * num_workers + worker_id

        # This is where we shard the dataset into work for each dataparallel rank.
        # Our unit of work is now a "row" of the dataset though, not a file.

        self._worker_id = worker_id

        # max_num_rows = (len(self._ds) // num_shards) * num_shards
        max_num_rows = len(self._ds)
        index_list = list(range(shard_id, max_num_rows, num_shards))

        if index_list == []:
            self._ds_fingerprint = None
            self._ds_min = 0
            self._ds_max = 0
        else:
            self._ds_fingerprint = hashlib.shake_128(str(index_list).encode()).hexdigest(4)
            self._ds_min = min(index_list)
            self._ds_max = max(index_list)

        subds = self._ds.select(index_list)
        self._subds = subds
        self.state = {"data_idx": 0}

        self._ds_length = len(self._subds)

        logger.info(
            f"Rank {self._process_rank}/{self._num_processes}, worker {worker_id} has "
            f"{self._ds_length}/{self._ds_total_length} rows | identifier={self._data_id}:{self._ds_fingerprint} "
            f"| range={self._ds_min}:{self._ds_max} | head={index_list[:3]} | tail={index_list[-3:]}"
        )

        return HuggingfaceDatasetIterator(
            ds=subds,
            data_signature=self._data_signature,
            data_id=self._data_id,
            return_data_id=self._return_data_id,
            fingerprint=self._ds_fingerprint,
            worker_id=worker_id,
            process_rank=self._process_rank,
            num_processes=self._num_processes,
            state=self.state,
        )

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def __len__(self):
        return self._ds_length


class HuggingfaceDatasetIterator:
    def __init__(
        self,
        ds,
        data_signature: dict[str, list[str] | str],
        data_id=None,
        return_data_id=None,
        fingerprint=None,
        worker_id=None,
        process_rank=None,
        num_processes=None,
        state={"data_idx": 0},
    ):
        self._ds = ds
        self._data_signature = data_signature
        self._data_id = data_id
        self._return_data_id = return_data_id
        self._ds_fingerprint = fingerprint
        self._worker_id = worker_id
        self._process_rank = process_rank
        self._num_processes = num_processes

        self._ds_iter = None
        self.state = state

    def __len__(self):
        return len(self._ds)

    def __next__(self):
        row = self._ds[self.state["data_idx"]]
        self.state["data_idx"] += 1

        # the data signature tells us what keys to extract from the row
        row = {k: row[k] for k in self._data_signature["keys"]}
        # then we attach the data_signature to the sample to support
        # heterogeneously sourced batches in the collate_fn
        row["data_signature"] = self._data_signature

        if self._return_data_id:
            row["data_id"] = self._data_id

        return row


class HuggingfaceCombinedDataset(IterableDataset):
    def __init__(self, datasets, seed, weights=None, data_telemetry=False):
        self._seed = seed
        self._datasets = datasets
        self._weights = weights
        self._data_telemetry = data_telemetry
        n_datasets = len(datasets)
        if weights is None:
            self._weights = [1 / n_datasets] * n_datasets
        else:
            self._weights = [w / sum(weights) for w in weights]

    def __iter__(self):
        return HuggingfaceCombinedDatasetIterator(self._datasets, self._seed, self._weights, self._data_telemetry)


class HuggingfaceCombinedDatasetIterator:
    def __init__(self, datasets, seed, weights, data_telemetry=False):
        self._datasets = [iter(el) for el in datasets]
        self._weights = weights
        self._rng = random.Random(seed)
        self._iter_ct = 0
        self._data_telemetry = data_telemetry

    def __next__(self):
        (dataset,) = self._rng.choices(self._datasets, weights=self._weights, k=1)
        self._iter_ct += 1

        # this is the very beginning of data telemetry
        if self._data_telemetry and self._iter_ct < 5:
            logger.info(
                f"Draw result i={self._iter_ct} for rank={dataset._process_rank}/{dataset._num_processes}, "
                f"worker={dataset._worker_id} | {dataset._data_id}:{dataset._ds_fingerprint}"
            )
        elif self._data_telemetry and self._iter_ct == 5:
            logger.info("Data telemetry off ...")

        return next(dataset)


class RandomTokensDataset(IterableDataset):
    def __init__(self, seed=233, vocab_size=1024, block_size=4096):
        # Debugging option
        generator = torch.Generator()
        generator.manual_seed(seed)
        self.data = torch.randint(
            0,
            vocab_size,
            (int(1e6), block_size),
            dtype=torch.int32,
            generator=generator,
        )
        self.idx = 0

    def __iter__(self):
        for entry in self.data[self.idx :]:
            self.idx += 1
            yield entry

    def state_dict(self):
        return {"idx": self.idx}

    def load_state_dict(self, state_dict):
        self.idx = state_dict["idx"]
