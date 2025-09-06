"""This script is an alternative option to using lm-eval to check adaptive exits. If you are new to this code base,
you should probably just use lm-eval.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, Union
import os
import sys
import contextlib

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import transformers
from transformers import GenerationConfig
from transformers.generation.utils import GenerateDecoderOnlyOutput

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


from recpre.raven_modeling_minimal import RavenForCausalLM, CausalLMOutputRecurrentLatents
from recpre.raven_modeling_minimal import PerIterationExitEvaluator, SupportedExitCriteria
from evaluate_raven.quick_checkpoint_eval import prepare_results


def update_huggingface_implementation(model):
    """This function selectively updates function implementations in the huggingface model."""
    import types
    
    # NOTE: edit this as needed in regards to your local changes to the model

    # for name, module in model.named_modules():
    #     if module.__class__.__name__ == "CausalSelfAttention":
    #         module.forward = types.MethodType(CausalSelfAttention.forward, module)
    model.generate = types.MethodType(RavenForCausalLM.generate, model)
    model.generate_with_adaptive_compute = types.MethodType(RavenForCausalLM.generate_with_adaptive_compute, model)
    model.forward = types.MethodType(RavenForCausalLM.forward, model)
    model.forward_with_adaptive_compute = types.MethodType(RavenForCausalLM.forward_with_adaptive_compute, model)
    model._prefill_with_varied_exit_steps = types.MethodType(RavenForCausalLM._prefill_with_varied_exit_steps, model)
    model._prep_generate_args = types.MethodType(RavenForCausalLM._prep_generate_args, model)
    model.prepare_inputs_for_generation = types.MethodType(RavenForCausalLM.prepare_inputs_for_generation, model)


class HuginnWrapper(HFLM):
    """Wrapper for Huginn model using lm_eval, extending HFLM."""

    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        *,
        backend: Literal["default", "causal", "seq2seq"] = "default",
        criterion: Literal["entropy-diff", "latent-diff", "cosine", "minp-kl", "kl", "argmax-stability", "none"] = "entropy-diff",
        exit_threshold: Union[str, float, int] = "auto",
        continuous_compute: bool = False,
        exit_evaluator: Optional[PerIterationExitEvaluator] = None,
        prefill_with_varied_exit_steps: bool = False,
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
            **kwargs,
        )
        self.criterion = criterion
        self.exit_threshold = exit_threshold
        self.continuous_compute = continuous_compute
        self.exit_evaluator = exit_evaluator
        self.prefill_with_varied_exit_steps = prefill_with_varied_exit_steps
        update_huggingface_implementation(self.model)
        self.compute_steps = []

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # The generation configs is only used by the custom generate function call,
        # whereas the standard generate function call uses these args directly passed in.
        # So we need to pass both, and have the dispatching generate function call decide which to use.
        generation_config = GenerationConfig(
            max_length=max_length,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
            return_dict_in_generate=True,
        )
        if (self.criterion == SupportedExitCriteria.NONE.value or self.exit_threshold == "none") and self.exit_evaluator is None:
            output = super()._model_generate(
                context,
                max_length,
                stop,
                generation_config=generation_config,
                continuous_compute=self.continuous_compute,
                **generation_kwargs,
            )
        else:
            output = super()._model_generate(
                context,
                max_length,
                stop,
                generation_config=generation_config,
                criterion=self.criterion,
                exit_threshold=self.exit_threshold,
                continuous_compute=self.continuous_compute,
                exit_evaluator=self.exit_evaluator,
                prefill_with_varied_exit_steps=self.prefill_with_varied_exit_steps,
                **generation_kwargs,
            )
        # Capture compute_steps if available
        if isinstance(output, GenerateDecoderOnlyOutput):
            compute_steps = [[] for _ in range(self.batch_size)]  # type: ignore
            if output.scores is not None:
                for s in output.scores:
                    for i in range(self.batch_size):  # type: ignore
                        compute_steps[i].append(s[0][i])
                self.compute_steps.append(compute_steps)
            return output.sequences
        return output

    def _model_call(self, inps, attn_mask=None, labels=None):
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :param attn_mask: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :param labels: torch.Tensor, optional
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)]. Only passed
            (and must be passed) if self.AUTO_MODEL_CLASS is transformers.AutoModelForSeq2SeqLM
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert transformers.AutoModelForSeq2SeqLM == self.AUTO_MODEL_CLASS
                output = self.model(input_ids=inps, attention_mask=attn_mask, labels=labels)
            else:
                assert transformers.AutoModelForCausalLM == self.AUTO_MODEL_CLASS
                # inject our custom exit evaluator into the model
                if self.exit_evaluator is not None:
                    output = self.model.forward_with_adaptive_compute(inps, exit_evaluator=self.exit_evaluator)
                else:
                    output = self.model(inps, exit_evaluator=self.exit_evaluator)
            if (
                isinstance(output, CausalLMOutputRecurrentLatents)
                and output.stats is not None
                and "compute_steps" in output.stats
            ):
                self.compute_steps.append(output.stats["compute_steps"])
            return output.logits


def evaluate_tasks(
    tasks = ["gsm8k"],
    model: Union[str, RavenForCausalLM] = "tomg-group-umd/huginn-0125",
    device="cuda",
    batch_size=1,
    num_fewshot=5,
    limit=None,
    criterion: Literal["entropy-diff", "latent-diff", "minp-kl", "argmax-stability", "none"] = "entropy-diff",
    exit_threshold: Union[str, float, int] = "auto",
    num_steps=32,
    lookup_strategy="full",
    exit_evaluator: Optional[PerIterationExitEvaluator] = None,
    continuous_compute=False,
    max_length=2048,
    output_path: Optional[str] = None,
    prefill_with_varied_exit_steps: bool = False,
    gen_kwargs: Optional[str] = "",
    **kwargs,
):
    model_name = model if isinstance(model, str) else "custom_model"
    if gen_kwargs is None:
        gen_kwargs = ""

    config_args = {
        "tasks": tasks,
        "model_name": model_name,
        "device": device,
        "batch_size": batch_size,
        "max_length": max_length,
        "num_fewshot": num_fewshot,
        "limit": limit,
        "criterion": criterion,
        "exit_threshold": exit_threshold,
        "num_steps": num_steps,
        "lookup_strategy": lookup_strategy,
        "exit_evaluator": exit_evaluator.__class__.__name__ if exit_evaluator is not None else None,
        "prefill_with_varied_exit_steps": prefill_with_varied_exit_steps,
        "gen_kwargs": gen_kwargs,
        **kwargs,
    }

    print(f"Evaluating {model_name} on {tasks} with config: {config_args}")
    model_wrapper = HuginnWrapper(
        pretrained=model,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        trust_remote_code=True,
        dtype="bfloat16",
        criterion=criterion,
        exit_threshold=exit_threshold,
        continuous_compute=continuous_compute,
        exit_evaluator=exit_evaluator,
        prefill_with_varied_exit_steps=prefill_with_varied_exit_steps,
    )
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        gen_kwargs=f"num_steps={num_steps},cache_lookup_strategy={lookup_strategy}," + gen_kwargs,
        **kwargs,
    )

    if results is not None:
        results["config_args"] = config_args
        # Add avg_compute_steps to results if available
        if hasattr(model_wrapper, "compute_steps") and model_wrapper.compute_steps:
            results["compute_steps"] = model_wrapper.compute_steps

        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%dT%H-%M-%S.%f")
        model_pathname = model_name.replace("/", "__")
        directory_path = Path(output_path, model_pathname) if output_path is not None else Path(model_pathname)
        os.makedirs(directory_path, exist_ok=True)
        prepare_results(results, Path(directory_path, f"hf_eval_results_{timestamp}.json"))
    return results


if __name__ == "__main__":
    import argparse
    from lm_eval.__main__ import setup_parser
    parser: argparse.ArgumentParser = setup_parser()
    parser.description = "lm_eval wrapper for evaluating Huginn on a task with adaptive compute."

    # drop args that are not supported by our script
    drop_args = [("model", "m"), ("model_args", "a"), "show_config", "include_path", "wandb_args", "hf_hub_log_args", "trust_remote_code"]
    for drop_arg in drop_args:
        if isinstance(drop_arg, tuple):
            drop_arg, drop_arg_short = drop_arg
        else:
            drop_arg_short = None
        parser._remove_action(parser._option_string_actions[f"--{drop_arg}"])
        parser._option_string_actions.pop(f"--{drop_arg}")
        if drop_arg_short is not None:
            parser._option_string_actions.pop(f"-{drop_arg_short}")

    parser.add_argument(
        "--criterion",
        type=str,
        default=SupportedExitCriteria.ENTROPY_DIFF.value,
        choices=[c.value for c in SupportedExitCriteria],
        help="Criterion for adaptive compute. Pass `none` to disable adaptive compute.",
    )
    parser.add_argument(
        "--exit_threshold",
        type=str,
        default="auto",
        help="Exit threshold for adaptive compute. Pass `none` to disable adaptive compute.",
    )
    parser.add_argument("--num_steps", type=int, default=32, help="Number of steps for generation")
    parser.add_argument(
        "--lookup_strategy",
        type=str,
        default="full",
        help="Lookup strategy for caching, also supports values like `compression-s4`",
    )
    parser.add_argument(
        "--continuous_compute", type=bool, default=False, help="Continuous compute"
    )
    parser.add_argument("--prefill_with_varied_exit_steps", type=bool, default=False, help="Prefill with varied exit steps")

    # remove log_samples and re-register it with a different default value
    parser._remove_action(parser._option_string_actions["--log_samples"])
    parser._option_string_actions.pop("--log_samples")
    parser._option_string_actions.pop("-s")
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=True,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis.",
    )

    args = parser.parse_args()

    if args.batch_size != 1 and args.criterion != SupportedExitCriteria.NONE.value:
        print(f"WARNING: please run adaptive compute with batch_size=1 if accurate results are desired.")

    # Try to convert exit_threshold to float if it's numeric
    if args.exit_threshold != "auto" and args.exit_threshold != "none":
        with contextlib.suppress(ValueError):
            args.exit_threshold = float(args.exit_threshold)  # Keep as string if not convertible to float

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = args.seed
    delattr(args, "seed")

    results = evaluate_tasks(
        model="tomg-group-umd/huginn-0125",
        random_seed=seeds[0],
        numpy_random_seed=seeds[1],
        torch_random_seed=seeds[2],
        fewshot_random_seed=seeds[3],
        **vars(args)
    )
