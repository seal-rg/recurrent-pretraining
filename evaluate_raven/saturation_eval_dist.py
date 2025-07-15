# Brute forcing the saturation eval
"""Use this file only if you want to replicate the saturation chart from the v3 version of the paper.
Do not use this for normal evals, it would be too wasteful."""

import torch
from jsonargparse import CLI

device = torch.device("cuda:0")
import sys
from pathlib import Path

wd = Path.cwd().parent
sys.path.append(str(wd))
import litgpt  # noqa: F401

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "0"
import datasets

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True  # type: ignore


def saturation_eval(
    task: str = "ifeval", num_steps: int = 32, subsampling=1, num_fewshot=0, device=device, limit=None, use_chat=False
):
    acc_storage = {}

    print(f"Testing {task} with {num_steps} recurrence steps.")
    model = HFLM(
        pretrained="tomg-group-umd/huginn-0125",
        device=device,  # type: ignore
        dtype="bfloat16",
        max_length=4096,
        trust_remote_code=False,
        batch_size=1 if task == "ifeval" and num_steps > 16 else "auto",
    )  # unfortunately forced to reload otherwise batch size tester does not rerun

    model._model.config.mean_recurrence = num_steps  # ignored if passed via model_args to lm-eval # type: ignore
    results: dict = evaluator.simple_evaluate(  # type: ignore
        model=model,
        tasks=[task],
        num_fewshot=num_fewshot,
        device=device,  # type: ignore
        limit=limit,
        batch_size=1 if task == "ifeval" and num_steps > 16 else "auto",
        confirm_run_unsafe_code=True,
        apply_chat_template=use_chat,
        system_instruction="You are a helpful assistant." if use_chat else None,
        fewshot_as_multiturn=use_chat,
    )
    print(f"Result is {results['results']}")
    if "gsm8k" in task:
        acc_storage[num_steps] = [
            sample["exact_match"] for sample in results["samples"][task] if sample["filter"] == "flexible-extract"
        ]
    elif "mmlu" in task or "blimp" in task:
        acc_storage[num_steps] = [sample["acc"] for key in results["samples"] for sample in results["samples"][key]]
    elif "mastermind" in task:
        acc_storage[num_steps] = [sample["acc"] for sample in results["samples"]["mastermind_46_hard"]]
    elif "acp_bench" in task:
        acc_storage[num_steps] = [sample["acc"] for sample in results["samples"]["acp_app_mcq"]]
    elif "bbh" in task or "musr" in task:
        acc_storage[num_steps] = [
            sample["acc_norm"] for key in results["samples"] for sample in results["samples"][key]
        ]
    elif "minerva" in task:
        acc_storage[num_steps] = [
            sample["math_verify"] for key in results["samples"] for sample in results["samples"][key]
        ]
    elif "ifeval" in task:
        acc_storage[num_steps] = [
            sample["prompt_level_loose_acc"] for sample in results["samples"][task]
        ]  # inst_level_loose_acc
    elif "humaneval" in task or "mbpp" in task:
        acc_storage[num_steps] = [sample["pass@1"] for sample in results["samples"][task]]
    elif "triviaqa" in task or "fld_default" in task:
        acc_storage[num_steps] = [sample["exact_match,remove_whitespace"] for sample in results["samples"][task]]
    else:
        try:
            acc_storage[num_steps] = [sample["acc_norm"] for sample in results["samples"][task]]
        except KeyError:
            acc_storage[num_steps] = [sample["acc"] for sample in results["samples"][task]]

    torch.save(acc_storage, f"saturation_{task}_{num_steps}_{num_fewshot}.pth")


if __name__ == "__main__":
    CLI(saturation_eval)
