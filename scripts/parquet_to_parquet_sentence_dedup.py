"""

Deduplication via datatrove to and from the parquet folder

"""

# current consts:
BASE_DIR = "/lustre/orion/csc569/scratch/smcleish/recurrent_data_sentence"
CACHE_DIR = f"{BASE_DIR}/test_cache"
FINAL_LOCATION = f"{BASE_DIR}/processed_dataset"
DATASET_STAGING = f"/lustre/orion/csc569/proj-shared/language_datasets/staging_dataset"
# DATASET_STAGING = f"/lustre/orion/csc569/scratch/smcleish/recurrent_data_temp"

DEDUP_STAGING = f"{BASE_DIR}/dedup_staging_dataset"
final_dataset_name = "dedup_v01"

TMP_STAGE_DIR = f"{CACHE_DIR}/tmp/minhash"
LOCAL_LOGS_FOLDER = f"{DEDUP_STAGING}/{final_dataset_name}/logs"
TOTAL_TASKS = 512
FINDER_WORKERS = 32
# TOTAL_TASKS = 2


from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils.typeshelper import Languages
from datatrove.pipeline.dedup import (
    SentenceDedupFilter,
    SentenceDedupSignature,
    SentenceFindDedups,
)
from datatrove.pipeline.dedup.sentence_dedup import SentDedupConfig
from datatrove.utils.typeshelper import Languages

# from datatrove.pipeline.tokens import TokensCounter

sent_dedup_config = SentDedupConfig(
    n_sentences=3,
    split_sentences=True,  # set to False to split on \n instead
    only_dedup_in_index=True,
    min_doc_words=50,
)

# this is the original data that we want to deduplicate
INPUT_READER = ParquetReader(DATASET_STAGING)

# 1. create a signature for each sentence in each doc
stage1 = SlurmPipelineExecutor(
    job_name="sent1",
    pipeline=[
        INPUT_READER,
        SentenceDedupSignature(
            output_folder=f"{TMP_STAGE_DIR}/signatures",
            config=sent_dedup_config,
            finder_workers=FINDER_WORKERS,
            language=Languages.english,
        ),
    ],
    tasks=TOTAL_TASKS,
    time="24:00:00",
    partition="batch",
    qos="normal",
    mem_per_cpu_gb=4,
    cpus_per_task=1,
    sbatch_args={"account": "CSC569", "nodes": 1, "ntasks-per-node": 32},
    logging_dir=f"{LOCAL_LOGS_FOLDER}/signatures",
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
)

# 2. reads all the signatures and loads them to check for duplicates.
stage2 = SlurmPipelineExecutor(
    job_name="sent2",
    pipeline=[
        SentenceFindDedups(
            data_folder=f"{TMP_STAGE_DIR}/signatures",
            output_folder=f"{TMP_STAGE_DIR}/buckets",
            config=sent_dedup_config,
        )
    ],
    tasks=FINDER_WORKERS,
    time="24:00:00",
    partition="batch",
    qos="normal",
    mem_per_cpu_gb=4,
    cpus_per_task=1,
    sbatch_args={"account": "CSC569", "nodes": 1, "ntasks-per-node": 32},
    logging_dir=f"{LOCAL_LOGS_FOLDER}/buckets",
    depends=stage1,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/buckets/slurm_logs",
)

# 3. reads a document pipeline and removes duplicated sentences found before
stage3 = SlurmPipelineExecutor(
    job_name="sent3",
    pipeline=[
        INPUT_READER,
        SentenceDedupFilter(
            data_folder=f"{TMP_STAGE_DIR}/buckets", config=sent_dedup_config
        ),
        ParquetWriter(
            output_folder=f"{DEDUP_STAGING}/{final_dataset_name}/deduplicated_output"
        ),
    ],
    tasks=TOTAL_TASKS,
    time="24:00:00",
    partition="batch",
    qos="normal",
    mem_per_cpu_gb=4,
    cpus_per_task=1,
    sbatch_args={"account": "CSC569", "nodes": 1, "ntasks-per-node": 32},
    logging_dir=f"{LOCAL_LOGS_FOLDER}/filter",
    depends=stage2,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filter/slurm_logs",
)

stage3.run()
