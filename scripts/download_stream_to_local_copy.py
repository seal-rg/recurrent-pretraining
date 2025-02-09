import logging
from multiprocessing import cpu_count

from datasets import load_dataset, Dataset, disable_caching, IterableDataset

disable_caching()
have_enough_ram = True


def download_stream(dataset_name_or_path, target_folder, shuffle_streaming_shards=False, subset_size=None, num_proc=0):
    dataset: IterableDataset = load_dataset(dataset_name_or_path, split="train", streaming=True, cache_dir=None)
    if shuffle_streaming_shards:
        dataset = dataset.shuffle(buffer_size=2**14, seed=233)
    if subset_size is not None:
        dataset = dataset.take(int(subset_size))

    # stream to disk
    ds_on_disk = Dataset.from_generator(dataset.__iter__, num_proc=num_proc, keep_in_memory=have_enough_ram)
    # finalize storage
    ds_on_disk.save_to_disk(target_folder, max_shard_size="4096MB", num_proc=num_proc)


def prepare(
    dataset_name_or_path: str = "HuggingFaceFW/fineweb-edu",
    target_folder: str = ".",
    subset_size: int = None,
    shuffle_streaming_shards: bool = False,
    num_proc: int = cpu_count(),
) -> None:
    """Download stream dataset because the HF streaming API is terrible"""

    logging.info(f"Running with num_proc={num_proc} on a machine with {cpu_count()} visible cpus.")

    download_stream(
        dataset_name_or_path=dataset_name_or_path,
        target_folder=target_folder,
        shuffle_streaming_shards=shuffle_streaming_shards,
        subset_size=subset_size,
        num_proc=num_proc,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
