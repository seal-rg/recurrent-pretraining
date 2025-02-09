"""

Deduplication via datatrove to and from the parquet folder

"""

# current consts:
BASE_DIR = "/lustre/orion/csc569/scratch/jgeiping/data"
CACHE_DIR = f"{BASE_DIR}/test_cache"
FINAL_LOCATION = f"{BASE_DIR}/processed_dataset"
DATASET_STAGING = f"{BASE_DIR}/staging_dataset"

DEDUP_STAGING = f"{BASE_DIR}/dedup_staging_dataset"
final_dataset_name = "dedup_v01"

TMP_STAGE_DIR = f"{CACHE_DIR}/tmp/minhash"
LOCAL_LOGS_FOLDER = f"{DEDUP_STAGING}/{final_dataset_name}/logs"
TOTAL_TASKS = 512


from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import (
    MinhashConfig,
    MinhashDedupBuckets,
    MinhashDedupCluster,
    MinhashDedupFilter,
)
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers.parquet import ParquetWriter
from datatrove.utils.hashing import HashConfig
from datatrove.utils.typeshelper import Languages


# you can also change ngrams or the number of buckets and their size here
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)  # better precision -> fewer false positives (collisions)


# this is the original data that we want to deduplicate
INPUT_READER = ParquetReader(DATASET_STAGING)

# stage 1 computes minhash signatures for each task (each task gets a set of files)
stage1 = SlurmPipelineExecutor(
    job_name="mh1",
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{TMP_STAGE_DIR}/signatures", config=minhash_config, language=Languages.english
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

# stage 2 finds matches between signatures in each bucket
stage2 = SlurmPipelineExecutor(
    job_name="mh2",
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{TMP_STAGE_DIR}/signatures",
            output_folder=f"{TMP_STAGE_DIR}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
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

# stage 3 creates clusters of duplicates using the results from all buckets
stage3 = SlurmPipelineExecutor(
    job_name="mh3",
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{TMP_STAGE_DIR}/buckets",
            output_folder=f"{TMP_STAGE_DIR}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,
    time="24:00:00",
    partition="batch",
    qos="normal",
    sbatch_args={"account": "CSC569", "nodes": 1, "ntasks-per-node": 1},
    logging_dir=f"{LOCAL_LOGS_FOLDER}/clusters",
    mem_per_cpu_gb=70,
    cpus_per_task=2,
    depends=stage2,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/clusters/slurm_logs",
)

# stage 4 reads the original input data and removes all but 1 sample per duplicate cluster
# the data must match exactly stage 1, so number of tasks and the input source must be the same
stage4 = SlurmPipelineExecutor(
    job_name="mh4",
    pipeline=[
        INPUT_READER,
        # TokensCounter(),  # nice way to see how many tokens we had before and after deduplication
        MinhashDedupFilter(input_folder=f"{TMP_STAGE_DIR}/remove_ids"),
        ParquetWriter(output_folder=f"{DEDUP_STAGING}/{final_dataset_name}/deduplicated_output"),
    ],
    tasks=TOTAL_TASKS,
    time="24:00:00",
    partition="batch",
    qos="normal",
    mem_per_cpu_gb=4,
    cpus_per_task=1,
    sbatch_args={"account": "CSC569", "nodes": 1, "ntasks-per-node": 32},
    logging_dir=f"{LOCAL_LOGS_FOLDER}/filter",
    depends=stage3,
    slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/filter/slurm_logs",
)

stage4.run()
