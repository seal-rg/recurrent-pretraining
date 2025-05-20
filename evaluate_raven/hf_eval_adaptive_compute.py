from pathlib import Path
from typing import Literal, Optional, Union
import os
import sys

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transformers
from transformers import GenerationConfig
import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria

from recpre.raven_modeling_minimal import CausalSelfAttention, RavenForCausalLM
from evaluate_raven.quick_checkpoint_eval import prepare_results


def update_huggingface_implementation(model):
    """This function selectively updates function implementations in the huggingface model."""
    import types
    # for name, module in model.named_modules():
    #     if module.__class__.__name__ == "CausalSelfAttention":
    #         module.forward = types.MethodType(CausalSelfAttention.forward, module)
    model.generate = types.MethodType(RavenForCausalLM.generate, model)
    model.generate_with_adaptive_compute = types.MethodType(RavenForCausalLM.generate_with_adaptive_compute, model)


class HuginnWrapper(HFLM):
    """Wrapper for Huginn model using lm_eval, extending HFLM."""

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        *,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        criterion: Optional[Literal["entropy-diff", "latent-diff", "minp-kl", "argmax-stability"]] = "entropy-diff",
        exit_threshold: Optional[Union[str, float, int]] = "auto",
        lookup_strategy: str = "full",
        continuous_compute: bool = False,
        latent_dampening: bool = False,
        # override whether the model should be treated as decoder-only (causal) or encoder-decoder (seq2seq)
        revision: Optional[str] = "main",
        subfolder: Optional[str] = None,
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ] = None,
        truncation: Optional[bool] = False,
        logits_cache: bool = True,
        max_length: Optional[int] = None,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 64,
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        add_bos_token: Optional[bool] = False,
        prefix_token_id: Optional[int] = None,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        parallelize: Optional[bool] = False,
        max_memory_per_gpu: Optional[Union[int, str]] = None,
        max_cpu_memory: Optional[Union[int, str]] = None,
        offload_folder: Optional[Union[str, os.PathLike]] = "./offload",
        # PEFT, delta weights and quantization options
        peft: Optional[str] = None,
        delta: Optional[str] = None,
        autogptq: Optional[Union[bool, str]] = False,
        gptqmodel: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained,
            backend,
            revision,
            subfolder,
            tokenizer,
            truncation,
            logits_cache,
            max_length,
            device,
            dtype,
            batch_size,
            max_batch_size,
            trust_remote_code,
            use_fast_tokenizer,
            add_bos_token,
            prefix_token_id,
            parallelize,
            max_memory_per_gpu,
            max_cpu_memory,
            offload_folder,
            peft,
            delta,
            autogptq,
            gptqmodel,
            **kwargs
        )
        self.criterion = criterion
        self.exit_threshold = exit_threshold
        self.lookup_strategy = lookup_strategy
        self.continuous_compute = continuous_compute
        self.latent_dampening = latent_dampening
        update_huggingface_implementation(self.model)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # The generation configs is only used by the custom generate function call, 
        # whereas the standard generate function call uses these args directly passed in.
        # So we need to pass both, and have the dispatching generate function call decide which to use.
        generation_config = GenerationConfig(
            max_length=max_length,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return super()._model_generate(
            context,
            max_length,
            stop,
            generation_config=generation_config,
            criterion=self.criterion, 
            exit_threshold=self.exit_threshold, 
            cache_kwargs={"lookup_strategy": self.lookup_strategy},
            continuous_compute=self.continuous_compute,
            latent_dampening=self.latent_dampening,
            **generation_kwargs
        )


def evaluate_single_task(
    task_name="gsm8k",
    model_name="tomg-group-umd/huginn-0125",
    device="cuda",
    batch_size=16,
    num_fewshot=5,
    limit=None,
    criterion: Optional[Literal["entropy-diff", "latent-diff", "minp-kl", "argmax-stability"]] = "entropy-diff",
    exit_threshold: Optional[Union[str, float, int]] = "auto",
    num_steps=32,
    lookup_strategy="full",
    continuous_compute=False,
    latent_dampening=False,
):
    config_args = {
        "task_name": task_name,
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "criterion": criterion,
        "exit_threshold": exit_threshold,
        "num_steps": num_steps,
        "lookup_strategy": lookup_strategy,
    }

    print(f"Evaluating {model_name} on {task_name} with config: {config_args}")
    model = HuginnWrapper(
        pretrained=model_name,
        device=device,
        batch_size=batch_size,
        trust_remote_code=True,
        dtype="bfloat16",
        criterion=criterion,
        exit_threshold=exit_threshold,
        lookup_strategy=lookup_strategy,
        continuous_compute=continuous_compute,
        latent_dampening=latent_dampening,
    )
    results = evaluator.simple_evaluate(
        model=model,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=limit,
        gen_kwargs=f"num_steps={num_steps}",
    )
    
    if results is not None:
        results["config_args"] = config_args
        prepare_results(results, Path(f"{task_name}_results.json"))
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model on a task with adaptive compute.")
    parser.add_argument("--task-name", dest="task_name", type=str, default="gsm8k", help="Task to evaluate on")
    parser.add_argument("--model-name", dest="model_name", type=str, default="tomg-group-umd/huginn-0125", help="Model to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda, cpu)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--num-fewshot", dest="num_fewshot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples to evaluate")
    parser.add_argument("--criterion", type=str, default="entropy-diff", 
                        choices=["entropy-diff", "latent-diff", "minp-kl", "argmax-stability", "none"],
                        help="Criterion for adaptive compute. Pass `none` to disable adaptive compute.")
    parser.add_argument("--exit-threshold", dest="exit_threshold", type=str, default="auto",
                        help="Exit threshold for adaptive compute. Pass `none` to disable adaptive compute.")
    parser.add_argument("--num-steps", dest="num_steps", type=int, default=32, help="Number of steps for generation")
    parser.add_argument("--lookup-strategy", dest="lookup_strategy", type=str, default="full", 
                        help="Lookup strategy for caching, also supports values like `compression-s4`")
    parser.add_argument("--continuous-compute", dest="continuous_compute", type=bool, default=False, help="Continuous compute")
    parser.add_argument("--latent-dampening", dest="latent_dampening", type=bool, default=False, help="Latent dampening")

    args = parser.parse_args()

    # Convert 'none' to None for criterion
    criterion = None if args.criterion == "none" else args.criterion

    # Try to convert exit_threshold to float if it's numeric
    exit_threshold = None if args.exit_threshold == "none" else args.exit_threshold
    if exit_threshold != "auto" and exit_threshold is not None:
        try:
            exit_threshold = float(exit_threshold)
        except ValueError:
            pass  # Keep as string if not convertible to float

    results = evaluate_single_task(
        task_name=args.task_name,
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        criterion=criterion,
        exit_threshold=exit_threshold,
        num_steps=args.num_steps,
        lookup_strategy=args.lookup_strategy,
        continuous_compute=args.continuous_compute,
        latent_dampening=args.latent_dampening,
    )
