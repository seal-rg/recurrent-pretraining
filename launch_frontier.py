# Hand-made frontier launch solution. Note that this is based only on  publicly available information regarding the Frontier.


"""Specs:
* Callable from python
* Automate good settings for HPC
* Handle file transfer and environment unpacking on all nodes
* Check a fixed git commit [optional]
* Load all required modules automatically
* Handle interconnect
* Handle default job dependencies, split into a series of 2h jobs, based on singleton execution

Todo:
* Handle automatic dependencies, such as cooldown and eval
"""

import os
import socket
import getpass
import secrets
import subprocess
import time
import argparse
from typing import Optional

from dataclasses import dataclass


def frontier_max_minutes(num_nodes: int):
    if num_nodes > 184:
        return 720
    elif num_nodes > 92:
        return 360
    else:
        return 120


def load_standard_modules(rocm_version=None):
    assert rocm_version is not None, "Please provide a valid ROCm version."
    return f"""
echo $(date -u) "Loading modules"
rocm_version={rocm_version}
# Load modules
module load PrgEnv-gnu/8.5.0
module load rocm/$rocm_version
module load craype-accel-amd-gfx90a
module load gcc-native/12.3
module load cray-mpich/8.1.28
libfabric_path=/opt/cray/libfabric/1.15.2.0
module load miniforge3
"""


def get_comms_and_slingshot(
    installdir="${WRKSPC}/tiny_plugins_rccl", enable_net_gdr=True, debug_flags=False, rccl_algo=None
):
    # algo can be TREE or RING (or None)
    # net_gdr is twice as fast on frontier, but may lead to hangs [...]
    # ENV variables are also documented in https://www.olcf.ornl.gov/wp-content/uploads/2021/04/HPE-Cray-MPI-Update-nfr-presented.pdf # noqa
    if not installdir.endswith("/lib"):
        installdir = installdir + "/lib"
    return f"""
### MPI
export MPICH_GPU_SUPPORT_ENABLED=0
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:${{CRAY_MPICH_ROOTDIR}}/gtl/lib"
### AMD GPU
export HSA_FORCE_FINE_GRAIN_PCIE=1
### Slingshot
export LD_LIBRARY_PATH="${{LD_LIBRARY_PATH}}:{installdir}"
export FI_CXI_ATS=0
{"export NCCL_NET_GDR_LEVEL=3" if enable_net_gdr else ""}
export NCCL_SOCKET_IFNAME=hsn
{"export NCCL_CHECKS_DISABLE=1" if not debug_flags else ""}
export {"NCCL_DEBUG_SUBSYS=INIT,COLL NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$(pwd)/rccl.%h.%p.log" if debug_flags else "NCCL_DEBUG=VERSION"}
{f"export NCCL_ALGO={rccl_algo}" if rccl_algo is not None else ""}
export NCCL_CROSS_NIC=1
"""  # noqa


def activate_env(env_path=r"${WRKSPC}/frontier_conda"):
    return f"""
source deactivate > /dev/null 2>&1
source activate {env_path}
echo $(date -u) "Activated environment on ${{SLURM_PROCID}}"
"""


def compress_working_dir():
    """Can also do this manually via
    tar -czf lit-gpt-compressed.tar.gz --transform="s|^|lit-gpt-dev/|" -T <(git ls-files --cached --others --exclude-standard)
    """
    git_ls = ["git", "ls-files", "--cached", "--others", "--exclude-standard"]
    files_in_repo = subprocess.Popen(git_ls, stdout=subprocess.PIPE)

    # Compress files into a tar.gz archive using tar
    package_dir = os.getenv("WRKSPC", os.environ["HOME"])
    compressed_repo = f"{package_dir}/lit-gpt-dev.tar.gz"
    tar_command = ["tar", "-czf", compressed_repo, "--transform=s|^|lit-gpt-dev/|", "-T", "-"]
    subprocess.run(tar_command, stdin=files_in_repo.stdout, text=True, check=True)
    print(f"The current working code was compressed to {compressed_repo}. Further changes will not affect the run!")
    return compressed_repo


def cast_archives(archives=["frontier_env_packed.tar.gz", "repo_compressed.tar.gz", "tiny_plugins_rccl.tar.gz"]):
    instruction = ""
    if any(arch is not None for arch in archives):
        instruction += f'echo $(date -u) "Copying {",".join(filter(None, archives))} to each node" \n'
        for archive in archives:
            if archive is not None:
                if not archive.endswith(".tar.gz"):
                    raise ValueError(f"Invalid environment archive path {archive} provided.")
                dirname, filename = os.path.split(archive)
                instruction += f"sbcast -pf {dirname}/{filename} /mnt/bb/${{USER}}/{filename} \n"

        instruction += """  
if [[ ! "$?" == "0" ]]; then
    echo "SBCAST failed!"
    exit 1
fi"""
    return instruction


def unpack_archives(archives, barrier=True, skip_if_existing=False):
    instruction = ""
    if any(arch is not None for arch in archives):
        instruction = "if [[ $LOCAL_RANK == 0 ]]; then \n"
        instruction += '    echo $(date -u) "Unpacking archives on rank ${SLURM_PROCID}..." \n'
        for archive in archives:
            if archive is not None:
                name = os.path.basename(archive).split(".tar.gz")[0]
                if skip_if_existing:
                    instruction += f"""
    if [[ ! -d /mnt/bb/${{USER}}/{name} ]]; then
        tar -xzf {archive} -C /mnt/bb/${{USER}}
    fi"""
                else:
                    instruction += f"    mkdir -p /mnt/bb/${{USER}}/{name} \n"
                    instruction += f"    tar -xzf {archive} -C /mnt/bb/${{USER}}/{name} \n"
        instruction += "fi \n"
        if barrier:
            instruction += bash_barrier(interval=5)
    return instruction


def bash_barrier(interval=5):
    barrier_file = f"barrier_{secrets.token_urlsafe(8)}"
    return f"""
if [[ $LOCAL_RANK == 0 ]]; then
    touch /mnt/bb/${{USER}}/{barrier_file}
fi
while [ ! -f /mnt/bb/${{USER}}/{barrier_file} ]
do
    sleep {interval}
done

"""


def set_generic_env_flags(
    run_name="debug-run",
    gpus_per_node=8,
    master_port=None,
    debug_flags_python=True,
    python_fault_handler=False,
    debug_flags_interconnect=False,
    host_on_rank_zero=True,
    output_dir="$(pwd)/output/$RUN_NAME",
):
    if host_on_rank_zero:
        master_address = "$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)"
    else:
        master_address = "$(hostname)"

    return f"""
export RUN_NAME="{run_name}"
NNODES=$SLURM_JOB_NUM_NODES
export MASTER_ADDR={master_address}
{f"export MASTER_PORT={master_port}" if master_port is not None else ""}
export WORLD_SIZE=$(( NNODES * {gpus_per_node} )) 
# debugging flags (optional)
{"export LOGLEVEL=INFO" if debug_flags_python else ""}
{"export PYTHONFAULTHANDLER=1" if python_fault_handler else ""}
{"export NCCL_DEBUG=WARN" if debug_flags_interconnect else ""}
{"export FI_LOG_LEVEL=warn" if debug_flags_interconnect else ""}
# frontier specific:
export OMP_NUM_THREADS=7 
# Dirs
export OUTPUT_DIR={output_dir}
export LOGDIR=${{OUTPUT_DIR}}/logs
mkdir -p $LOGDIR

echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "Logging to $LOGDIR"
# Compile
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE=1 # need to keep this if we don't recompile the whole shebang
export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=1
# Remove ipv4 warnings:
export TORCH_CPP_LOG_LEVEL=ERROR
"""


def set_internet_env_variables():
    return """
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy="http://proxy.ccs.ornl.gov:3128/"
export HTTPS_PROXY="http://proxy.ccs.ornl.gov:3128/"
"""


def assemble_sbatch_file(
    output_dir: str,
    run_name="debug-run",
    python_invocation="pretrain_umd/train.py",
    nodes=8,
    budget_minutes=120,
    rccl_installdir="${WRKSPC}/tiny_plugins_rccl",
    rocm_version=None,
    environment: str = "${WRKSPC}/frontier_conda_62",
    email=None,
    gpus_per_node=8,
    gpu_bind=False,  # Cannot use with lightning.fabric - works with axonn though!
    repetitions=1,
    dependency=None,
    cast_working_dirs=True,
    overwrite_env_flags=None,
    enable_internet=False,
    enable_net_gdr=True,
    inductor_cache=False,
):
    hours = budget_minutes // 60
    minutes = budget_minutes - hours * 60
    # Find a free socket:
    sock = socket.socket()
    sock.bind(("", 0))
    free_socket_frontier = sock.getsockname()[1]
    sock.close()
    # Prealloc logfile and output folder
    logdir = f"{output_dir}/logs"
    os.makedirs(logdir, exist_ok=True)
    # Find archives
    if environment.endswith(".tar.gz"):
        env_packed = environment
        environment = f"/mnt/bb/${{USER}}/{os.path.basename(environment).split('.tar.gz')[0]}"
    else:
        env_packed = None
    cwd_compressed = compress_working_dir() if cast_working_dirs else None
    if rccl_installdir.endswith(".tar.gz"):
        rccl_compressed = rccl_installdir
        # get the last part of the path before the .tar.gz
        dir_name = os.path.basename(rccl_installdir).split(".tar.gz")[0]
        final_rccl_dir = f"/mnt/bb/${{USER}}/{dir_name}/lib"
    else:
        rccl_compressed = None
        final_rccl_dir = rccl_installdir
    if inductor_cache:
        if inductor_cache.endswith(".tar.gz"):
            cache_file = inductor_cache
            assert os.path.exists(os.path.expandvars(cache_file))
            filename = inductor_cache.split("/")[-1]
            basename = filename.split(".")[0]
            final_inductor_cache_dir = "/mnt/bb/${USER}/" + basename
        else:
            final_inductor_cache_dir = os.path.expandvars(inductor_cache)
    else:
        # default to use cache if no other cache is set
        cache_file = None
        final_inductor_cache_dir = "/lustre/orion/csc569/scratch/${USER}/inductor_cache"

    sbatch_file = rf"""#!/bin/bash
#SBATCH --account=csc569
#SBATCH --time={hours}:{minutes:02d}:00
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --constraint=nvme

#SBATCH --array=1-{repetitions}%1 
{f"#SBATCH --dependency={dependency}" if dependency is not None else ""}

#SBATCH --job-name={run_name}
#SBATCH --output={logdir}/%x_%A_%a.log
#SBATCH --error={logdir}/%x_%A_%a.log
#SBATCH --open-mode=append
{f"#SBATCH --mail-user={email}" if email is not None else ""}
{"#SBATCH --mail-type=FAIL,ARRAY_TASKS" if email is not None else ""}

echo $(date -u) "Preparing run..."
{load_standard_modules(rocm_version)}
{cast_archives(archives=[env_packed, cwd_compressed, rccl_compressed, cache_file])}
{get_comms_and_slingshot(final_rccl_dir, enable_net_gdr=enable_net_gdr) if rccl_installdir not in [None, ""] else ""}
{set_generic_env_flags(run_name=run_name, gpus_per_node=gpus_per_node, master_port=free_socket_frontier, output_dir=output_dir)}
{set_internet_env_variables() if enable_internet else ""}
{f"export {overwrite_env_flags}" if overwrite_env_flags not in [None, ""] else ""}

echo $(date -u) "Starting run..."
srun --unbuffered -l -N {nodes} -n {gpus_per_node * nodes} -c7 --ntasks-per-node={gpus_per_node} \
    --gpus-per-node={gpus_per_node} {"--gpus-per-task=1 --gpu-bind=closest" if gpu_bind else ""} \
    ${{OUTPUT_DIR}}/{run_name}_runner.sh
echo $(date -u) "Job execution finished."
"""  # noqa

    runner = rf"""#!/bin/bash
export LOCAL_RANK=${{SLURM_LOCALID}}
export RANK=${{SLURM_PROCID}}
export WORLD_SIZE=${{SLURM_NTASKS}}
export TORCHINDUCTOR_CACHE_DIR={final_inductor_cache_dir}
{unpack_archives(archives=[env_packed, cwd_compressed, rccl_compressed, cache_file], barrier=True)}
{activate_env(environment)}
wandb offline > /dev/null 2>&1
export PYTHONPATH={os.getcwd() if cwd_compressed is None else "/mnt/bb/${USER}/lit-gpt-dev"}
cd $PYTHONPATH

ulimit -n 131070
echo $(date -u) "Launching python on ${{SLURM_PROCID}}..."
python -u {python_invocation}
"""

    return sbatch_file, runner


def assemble_salloc_file(
    output_dir: str,
    rccl_installdir="${WRKSPC}/tiny_plugins_rccl",
    nodes=8,
    environment: str = "${WRKSPC}/frontier_conda_62",
    rocm_version=None,
    gpus_per_node=8,
    overwrite_env_flags=None,
    enable_internet=False,
):
    assert not environment.endswith(".tar.gz")
    assert not rccl_installdir.endswith(".tar.gz")

    # Prealloc logfile and output folder
    logdir = f"{output_dir}/logs"
    os.makedirs(logdir, exist_ok=True)
    # Find a free socket:
    sock = socket.socket()
    sock.bind(("", 0))
    free_socket_frontier = sock.getsockname()[1]

    return rf"""#!/bin/bash
echo $(date -u) "Preparing interactive session..."
{load_standard_modules(rocm_version)}
export PYTHONPATH={os.getcwd()}
{get_comms_and_slingshot(rccl_installdir)}
{set_generic_env_flags(run_name="shminteractive", gpus_per_node=gpus_per_node, master_port=free_socket_frontier, output_dir=output_dir)}
{set_internet_env_variables() if enable_internet else ""}
{activate_env(environment)}
{f"export {overwrite_env_flags}" if overwrite_env_flags not in [None, ""] else ""}
alias pythonAll="srun -u -l -N{nodes} -n{gpus_per_node * nodes} -c7 --ntasks-per-node={gpus_per_node} --gpus-per-node={gpus_per_node} python -u"
alias pythonOne="srun -u -l -N1 -n1 -c7 --ntasks-per-node=1 --gpus-per-node=1 python -u"
echo $(date -u) "Starting session..."
"""  # noqa


@dataclass
class SLURMLaunch:
    output_dir: str
    sub_output_dir_name: Optional[str] = None
    default_python_invocation: str = "pretrain_umd/train.py"
    nodes: int = 8
    budget_minutes: int = 120
    rccl_installdir: str = "${WRKPSC}/tiny_plugins_rccl/lib"
    rocm_version: str = "6.2.0"
    environment: str = "${WRKPSC}/frontier_conda_62"
    email: Optional[str] = None
    gpus_per_node: int = 8
    repetitions: Optional[int] = None
    dependency: Optional[str] = None
    cast_working_dirs: bool = False
    overwrite_env_flags: Optional[str] = None
    enable_internet: bool = False
    enable_net_gdr: bool = True
    inductor_cache: Optional[str] = None
    timeslot: int = 10080

    def minutes_to_jobs(self):
        minutes_in_job = min(min(frontier_max_minutes(self.nodes), self.timeslot), self.budget_minutes)
        if self.repetitions is None:
            computed_repetitions = 1 + self.budget_minutes // min(frontier_max_minutes(self.nodes), self.timeslot)
        else:
            computed_repetitions = self.repetitions
        assert computed_repetitions > 0, "Repetitions must be a positive integer."
        return minutes_in_job, computed_repetitions

    def get_output_dir(self, run_name):
        sub_output_dir_name = run_name if self.sub_output_dir_name is None else self.sub_output_dir_name

        if self.output_dir is None:
            output_dir = f"{os.getcwd()}/outputs/{sub_output_dir_name}"
        else:
            output_dir = f"{self.output_dir}/{sub_output_dir_name}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def sbatch_repr(self, run_name: str, python_invocation: str):
        minutes_in_job, computed_repetitions = self.minutes_to_jobs()

        sbatch_file, runner_file = assemble_sbatch_file(
            output_dir=self.get_output_dir(run_name),
            run_name=run_name,
            python_invocation=python_invocation,
            nodes=self.nodes,
            budget_minutes=minutes_in_job,
            rccl_installdir=self.rccl_installdir,
            rocm_version=self.rocm_version,
            environment=self.environment,
            email=self.email,
            gpus_per_node=self.gpus_per_node,
            overwrite_env_flags=self.overwrite_env_flags,
            repetitions=computed_repetitions,
            dependency=self.dependency,
            cast_working_dirs=self.cast_working_dirs,
            enable_internet=self.enable_internet,
            enable_net_gdr=self.enable_net_gdr,
            inductor_cache=self.inductor_cache,
        )
        return sbatch_file, runner_file

    def salloc_repr(self):
        salloc_file = assemble_salloc_file(
            output_dir=self.get_output_dir("interactive"),
            rccl_installdir=self.rccl_installdir,
            rocm_version=self.rocm_version,
            nodes=self.nodes,
            environment=self.environment,
            gpus_per_node=self.gpus_per_node,
            overwrite_env_flags=self.overwrite_env_flags,
            enable_internet=self.enable_internet,
        )
        return salloc_file

    def execute(
        self,
        run_name="debug-run",
        python_invocation="pretrain_umd/train.py",
        dryrun=True,
        debug_qos=True,
        extended_partition=False,
        use_chain=False,
    ):
        if dryrun:
            print(f"Would launch {1 + self.budget_minutes // frontier_max_minutes(self.nodes)} jobs")
            return
        if use_chain:
            return self.execute_chain(run_name, python_invocation, dryrun, debug_qos, extended_partition)
        else:
            return self.execute_array(run_name, python_invocation, dryrun, debug_qos, extended_partition)

    def execute_array(
        self,
        run_name="debug-run",
        python_invocation="pretrain_umd/train.py",
        dryrun=False,
        debug_qos=True,
        extended_partition=False,
    ):
        sbatch_file, runner_file = self.sbatch_repr(run_name, python_invocation)
        sbatch_file_name = f"{run_name}_launch.sbatch"
        sbatch_file_path = f"{self.get_output_dir(run_name)}/{sbatch_file_name}"
        with open(sbatch_file_path, "w") as file:
            file.write(sbatch_file)
        os.system(f"chmod +x {sbatch_file_path}")
        with open(f"{self.get_output_dir(run_name)}/{run_name}_runner.sh", "w") as file:
            file.write(runner_file)
        os.system(f"chmod +x {self.get_output_dir(run_name)}/{run_name}_runner.sh")
        print("Launch Specs are:")
        print(sbatch_file)
        print(runner_file)

        username = getpass.getuser()
        print(f"Preparing job as user {username} for launch from {socket.gethostname()} in 10 seconds...")
        print(f"This will allocate {self.nodes} nodes, so {self.nodes * self.gpus_per_node} GPUS in total.")

        if not dryrun:
            print(
                f"An array with {1 + self.budget_minutes // frontier_max_minutes(self.nodes)} jobs will be launched to SLURM, "
                f"with {self.dependency} dependencies."
            )
            output_status = subprocess.run(
                [
                    "/usr/bin/sbatch",
                    f"--qos={'debug' if debug_qos else 'normal'}",
                    f"--partition={'extended' if extended_partition else 'batch'}",
                    f"{sbatch_file_path}",
                ],
                capture_output=True,
            )
            if len(output_status.stderr) > 0:
                raise ValueError(output_status.stderr)
            process_id = output_status.stdout.decode("utf-8").split("batch job ")[1].split("\n")[0]
            print(f"Launched job array with process id {process_id}.")

        else:
            print(
                f"An array of {1 + self.budget_minutes // frontier_max_minutes(self.nodes)} jobs would be launched to SLURM "
                f", with {self.dependency} dependencies."
            )
            print(f"No jobs are launched now. You can inspect the sbatch file at {sbatch_file_path}.")

    def execute_chain(
        self,
        run_name: str,
        python_invocation: str,
        dryrun=False,
        debug_qos: bool = False,
        extended_partition: bool = False,
    ) -> Optional[str]:
        """Execute jobs in a dependency chain instead of an array."""
        num_jobs = 1 + self.budget_minutes // frontier_max_minutes(self.nodes)
        print(f"Launching {num_jobs} chained jobs...")
        output_dir = self.get_output_dir(run_name)
        current_job_id = None
        chain_id = None

        # Single loop for all jobs
        for i in range(num_jobs):
            # Update dependency after first job
            if current_job_id is not None:
                self.dependency = f"afterany:{current_job_id}"

            # Get base sbatch file and remove array directive
            sbatch_file, runner_file = self.sbatch_repr(run_name, python_invocation)
            sbatch_file = "\n".join(line for line in sbatch_file.split("\n") if not line.startswith("#SBATCH --array"))

            # Update output/error patterns
            o = f"#SBATCH --output={output_dir}/logs"
            e = f"#SBATCH --error={output_dir}/logs"
            if chain_id is None:
                sbatch_file = sbatch_file.replace(f"{o}/%x_%A_%a.log", f"{o}/%x_%j_step{i}_%j.log")
                sbatch_file = sbatch_file.replace(f"{e}/%x_%A_%a.log", f"{e}/%x_%j_step{i}_%j.log")
            else:
                sbatch_file = sbatch_file.replace(f"{o}/%x_%A_%a.log", f"{o}/%x_{chain_id}_step{i}_%j.log")
                sbatch_file = sbatch_file.replace(f"{e}/%x_%A_%a.log", f"{e}/%x_{chain_id}_step{i}_%j.log")

            # Write sbatch file
            sbatch_file_path = f"{output_dir}/{run_name}_launch_{i}.sbatch"
            with open(sbatch_file_path, "w") as file:
                file.write(sbatch_file)
            os.system(f"chmod +x {sbatch_file_path}")

            # Write runner file (only needed once)
            if i == 0:
                with open(f"{output_dir}/{run_name}_runner.sh", "w") as file:
                    file.write(runner_file)
                os.system(f"chmod +x {output_dir}/{run_name}_runner.sh")

            if dryrun:
                print(
                    f"Initial job of {num_jobs} jobs would be launched to SLURM , with {self.dependency} dependencies."
                )
                print(f"No jobs are launched now. You can inspect the sbatch file at {sbatch_file_path}.")
                return None

            # Submit job
            output_status = subprocess.run(
                [
                    "/usr/bin/sbatch",
                    f"--qos={'debug' if debug_qos else 'normal'}",
                    f"--partition={'extended' if extended_partition else 'batch'}",
                    f"{sbatch_file_path}",
                ],
                capture_output=True,
            )
            if len(output_status.stderr) > 0:
                print(f"Error launching job {i}: {output_status.stderr.decode('utf-8')}")
                raise ValueError(f"Launch failure at job submission {i}.")

            current_job_id = output_status.stdout.decode("utf-8").split("batch job ")[1].split("\n")[0]
            if chain_id is None:
                chain_id = current_job_id
                print(f"Launched initial job with ID {current_job_id}")
            else:
                print(f"Launched job {i} with ID {current_job_id}")
            time.sleep(1)

        return current_job_id

    def interact(self, debug_qos=True, extended_partition=False, budget_minutes=120):
        salloc_file = self.salloc_repr()
        salloc_file_name = "interactive_session_launcher.sh"
        salloc_file_path = f"{self.get_output_dir('interactive')}/{salloc_file_name}"
        with open(salloc_file_path, "w") as file:
            file.write(salloc_file)
        username = getpass.getuser()
        hours = budget_minutes // 60
        minutes = budget_minutes - hours * 60
        print(
            f"Launching interactive session for {username}"
            f" for launch from {socket.gethostname()} for {hours}:{minutes} ..."
        )
        print(f"This will allocate {self.nodes} nodes, so {self.nodes * self.gpus_per_node} GPUS in total.")
        print(salloc_file)
        allocation = [
            "--account=csc569",
            f"--time={hours:02d}:{minutes:02d}:00",
            f"--nodes={self.nodes}",
            f"--gres=gpu:{self.gpus_per_node}",
            "--job-name=shminteractive",
            # "--dependency=singleton", # multiple interactive sessions are ok
            f"--qos={'debug' if debug_qos else 'normal'}",
            f"--partition={'extended' if extended_partition else 'batch'}",
        ]
        srun_invocation = [
            "srun",
            "-N1",
            "-n1",
            "--interactive",
            "--preserve-env",
            "--pty",
        ]
        # Start the interactive session
        cmd = ["/usr/bin/salloc", *allocation, *srun_invocation, "bash", "--rcfile", salloc_file_path]
        print(" ".join(cmd))
        subprocess.Popen(cmd).wait()
        print("Interactive session terminated successfully.")


def parse_and_execute():
    """Parser, specificially turned to files that look like the pretrain_umd.train.py file.
    You can always replace this parse with your
    own construction of a SLURMLaunch object, to which you can pass your desired python invocation,
    or you can use --custom_invocation=debug_script.py or so to entirely overwrite the launcher, but other train
    files can be launched with --python_script=ds_optim.train.py

    Interactive debugging can be launched by setting --interactive
    """

    parser = argparse.ArgumentParser(description="Dispatch a particular launch onto frontier.")
    # Base
    parser.add_argument("--run_name", default="frontier-debug", type=str, help="Name that will be displayed in squeue")
    parser.add_argument("--uuid", action="store_true", help="Make run name unique by appending a uid.")
    parser.add_argument("--python_script", default="train.py", type=str, help="Pretrain script.")
    parser.add_argument("--config", default=None, type=str, help="Which config? If None, no config is passed.")
    parser.add_argument("--dryrun", action="store_true", help="The sbatch file is only written and can be modified.")

    parser.add_argument("--interactive", action="store_true", help="Debug Session.")
    # Environment and Reqs
    parser.add_argument(
        "--rccl_installdir",
        default="${WRKSPC}/aws-ofi-rccl_571",
        type=str,
        help="RCCL plugin location either packed or unpacked. Unpacked folders will be loaded directly. Packed will be moved.",
    )
    parser.add_argument("--rocm_version", default="5.7.1", type=str, help="ROCm version.")
    parser.add_argument(
        "--environment",
        default="${WRKSPC}/frontier_conda_571",
        type=str,
        help="Environment path, either packed or unpacked. Unpacked envs will be loaded directly. Packed envs will be moved.",
    )
    parser.add_argument("--cast_working_dirs", action="store_true", help="Whether to move the litgpt folder")
    # Job details
    parser.add_argument("--budget_minutes", default=120, type=int, help="Requested runtime in minutes")
    parser.add_argument("--budget_hours", default=0, type=int, help="Requested runtime in hours.")
    parser.add_argument("--budget_days", default=0, type=int, help="Requested runtime in days.")
    parser.add_argument("--nodes", default="1", type=int, help="Requested number of nodes.")
    # Optional
    # You can provide a particular output_dir name and sub_output dir name
    # the default is to place your output dir at os.getcwd()/output and to generate a unique subdir called run_name
    parser.add_argument("--enable_internet", action="store_true", help="Enable proxies to the internet.")
    parser.add_argument("--output_dir", default=None, type=str, help="The output dir.")
    parser.add_argument("--sub_output_dir_name", default=None, type=str, help="dir where all run-related files go")
    parser.add_argument("--extra_args", default="", type=str, help="Extra arguments to train.py as --arg=X")
    parser.add_argument("--email", default=None, type=str, help="Your email.")
    # Debugging
    parser.add_argument("--debug_qos", action="store_true", help="Launch onto debug queue.")
    parser.add_argument("--extended_partition", action="store_true", help="Launch onto extended queue.")
    parser.add_argument("--gpus_per_node", default=8, type=int, help="Requested number of GPUs per node.")
    parser.add_argument("--repetitions", default=None, type=int, help="Manual number of repetitions.")
    parser.add_argument("--custom_invocation", default=None, type=str, help="Use a completely custom python invocation")
    parser.add_argument("--overwrite_env", default=None, type=str, help="Overwrite env vars 'ENV_VAR=X ENV_VAR_2=Y'")
    parser.add_argument("--dependency", default=None, type=str, help="Specify dependency type.")
    parser.add_argument("--disable_net_gdr", action="store_true", help="Disable net gdr for NCCL.")
    parser.add_argument("--inductor_cache", default=None, type=str, help="Location of cache, either packed or unpacked")
    parser.add_argument("--use_chain", action="store_true", help="Use chained jobs instead of job array")
    parser.add_argument("--timeslot", type=int, default=10080, help="Optionally set a shorter timeslot (in minutes).")
    args = parser.parse_args()

    actual_budget_minutes = args.budget_minutes + 60 * args.budget_hours + 60 * 24 * args.budget_days
    # Define launch settings, environment and SLURM directives at construction time
    launch_object = SLURMLaunch(
        output_dir=args.output_dir,
        sub_output_dir_name=args.sub_output_dir_name,
        nodes=args.nodes,
        gpus_per_node=args.gpus_per_node,
        budget_minutes=actual_budget_minutes,
        rccl_installdir=args.rccl_installdir,
        rocm_version=args.rocm_version,
        environment=args.environment,
        email=args.email,
        overwrite_env_flags=args.overwrite_env,
        repetitions=args.repetitions,
        dependency=args.dependency,
        cast_working_dirs=args.cast_working_dirs,
        enable_internet=args.enable_internet,
        enable_net_gdr=(not args.disable_net_gdr),
        inductor_cache=args.inductor_cache,
        timeslot=args.timeslot,
    )

    if args.uuid:
        authkey = secrets.token_urlsafe(5)
        args.run_name = f"{args.run_name}_{authkey}"
        print(f"Assigned unique run name {args.run_name}")

    if args.custom_invocation is None:
        fully_assembled_invocation = (
            f"{args.python_script} "
            f"{f'--config={args.config} ' if args.config is not None else ''}"
            f"--run_name={args.run_name} --out_dir={launch_object.get_output_dir(args.run_name)} {args.extra_args}"
        )
    else:
        fully_assembled_invocation = args.custom_invocation

    if not args.interactive:
        # Execute a particular python command (here `fully_assembled_invocation`) with obj.execute()
        launch_object.execute(
            python_invocation=fully_assembled_invocation,
            run_name=args.run_name,
            dryrun=args.dryrun,
            debug_qos=args.debug_qos,
            extended_partition=args.extended_partition,
            use_chain=args.use_chain,
        )
    else:
        launch_object.interact(
            debug_qos=args.debug_qos, extended_partition=args.extended_partition, budget_minutes=actual_budget_minutes
        )


if __name__ == "__main__":
    parse_and_execute()
