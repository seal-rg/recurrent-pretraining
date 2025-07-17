# Step-aware lm-eval with command line interface and histogram generation
# Requires: pip install matplotlib

import numpy as np
import matplotlib.pyplot as plt
from lm_eval import simple_evaluate
from lm_eval.api.registry import get_model
import argparse
import json
from lm_eval.utils import make_table, handle_non_serializable
from lm_eval.loggers import EvaluationTracker
from pathlib import Path
import sys

####### oh I really stopped caring in this part
try:
    import recpre  # noqa: F401
except ModuleNotFoundError:
    try:
        wd = Path.cwd()
        sys.path.append(str(wd))
        import recpre  # noqa: F401
    except ModuleNotFoundError:
        wd = Path.cwd().parent
        sys.path.append(str(wd))
        import recpre  # noqa: F401
#########################


def evaluate_with_steps(model, tasks, model_args, **kwargs):
    """Run lm-eval + capture adaptive computation steps in one pass."""

    step_counts = []

    # Store original generate method from the underlying Huginn model
    huginn_model = model._model  # HFLM stores HF model in _model
    original_generate = huginn_model.generate

    def step_tracking_generate(*args, **gen_kwargs):
        # Ensure we get step info back
        model._model.generation_config.return_dict_in_generate = True

        # Call original generate
        output = original_generate(*args, **gen_kwargs)

        if not output.scores:
            raise ValueError("No scores recorded, is this an adaptive generation?")
        # Extract step counts
        for token_steps, _ in output.scores:
            step_counts.extend([t for t in token_steps if t > 0])  # 0 means it's a padded token

        return output.sequences

    # Monkey patch during evaluations
    huginn_model.generate = step_tracking_generate

    try:
        # Run normal lm-eval
        results = simple_evaluate(model, model_args=model_args, tasks=tasks, **kwargs)

        # Add step statistics to results
        step_stats = {
            "adaptive_avg_steps": float(np.mean(step_counts)),
            "adaptive_min_steps": int(np.min(step_counts)),
            "adaptive_max_steps": int(np.max(step_counts)),
            "adaptive_std_steps": float(np.std(step_counts)),
            "adaptive_total_tokens": len(step_counts),
        }

        # Add to each task's results
        if results is not None:  # None on other workers?
            for task_name in results["results"]:  # type: ignore
                results["results"][task_name].update(step_stats)  # type: ignore
    finally:
        # Restore original method
        huginn_model.generate = original_generate

    return results, step_counts


def save_step_histogram(step_counts, filename):
    """Save histogram of step counts."""
    plt.figure(figsize=(10, 6))
    plt.hist(step_counts, bins=max(10, max(step_counts) - min(step_counts) + 1), alpha=0.7, edgecolor="black")
    plt.xlabel("Number of Steps")
    plt.ylabel("Frequency")
    plt.title(
        f"Distribution of Adaptive Computation Steps\nMean: {np.mean(step_counts):.2f}, Std: {np.std(step_counts):.2f}"
    )
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def parse_gen_kwargs(gen_kwargs_str):
    """Parse gen_kwargs string like 'key1=val1,key2=val2' into dict."""
    if not gen_kwargs_str:
        return {}

    kwargs = {}
    for pair in gen_kwargs_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            # Try to convert to appropriate type
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.replace(".", "").replace("-", "").replace("e", "").isdigit():
                value = float(value) if "." in value or "e" in value.lower() else int(value)
            kwargs[key] = value
    return kwargs


# Usage script - replaces your command line call
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="lm-eval with adaptive computation step tracking")
    parser.add_argument("--model", default="hf", help="Model type")
    parser.add_argument("--model_args", required=True, help="Model arguments (comma-separated)")
    parser.add_argument("--tasks", required=True, help="Tasks to evaluate (comma-separated)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--output_path", help="Output directory")
    parser.add_argument("--apply_chat_template", action="store_true", help="Apply chat template")
    parser.add_argument("--system_instruction", help="System instruction")
    parser.add_argument("--fewshot_as_multiturn", action="store_true", help="Few-shot as multiturn")
    parser.add_argument("--gen_kwargs", help="Generation kwargs (comma-separated)")
    parser.add_argument("--limit", type=int, help="Limit number of examples")

    args = parser.parse_args()

    # Create model
    model = get_model(args.model).create_from_arg_string(args.model_args)

    # Parse tasks
    tasks = [task.strip() for task in args.tasks.split(",")]

    # Parse gen_kwargs
    gen_kwargs = parse_gen_kwargs(args.gen_kwargs)

    # Build evaluation kwargs
    eval_kwargs = {
        "batch_size": args.batch_size,
        "num_fewshot": args.num_fewshot,
        "gen_kwargs": gen_kwargs,
        "log_samples": False,
    }

    if args.apply_chat_template:
        eval_kwargs["apply_chat_template"] = True
    if args.system_instruction:
        eval_kwargs["system_instruction"] = args.system_instruction
    if args.fewshot_as_multiturn:
        eval_kwargs["fewshot_as_multiturn"] = True
    if args.limit:
        eval_kwargs["limit"] = args.limit

    evaluation_tracker = EvaluationTracker(output_path=args.output_path)
    eval_kwargs["evaluation_tracker"] = evaluation_tracker

    # Run evaluation
    results, step_counts = evaluate_with_steps(model=model, tasks=tasks, model_args=args.model_args, **eval_kwargs)

    # Print summary
    if results is not None:
        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:  # type: ignore
            print(make_table(results, "groups"))

    # Steal lm-eval printer:
    if results is not None:
        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        evaluation_tracker.save_results_aggregated(results=results, samples=None)  # type: ignore

        # Save histogram
        full_path = f"{args.output_path}/{evaluation_tracker.general_config_tracker.model_name_sanitized}"
        if full_path and step_counts:
            save_step_histogram(step_counts, f"{full_path}/step_histogram_{evaluation_tracker.date_id}.png")
        # Save all step counts in a single json
        if full_path:
            with open(f"{full_path}/step_counts_{evaluation_tracker.date_id}.json", "w") as f:
                json.dump(step_counts, f)


# To run (replaces your original command):
"""
CUDA_VISIBLE_DEVICES=2,3,4,5 accelerate launch evaluate_raven/record_steps_in_bench.py \
  --model hf \
  --model_args "pretrained=tomg-group-umd/huginn_swa_75_7_ema_0.9_merge,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32" \
  --tasks gsm8k_cot \
  --batch_size 1 \
  --num_fewshot 8 \
  --output_path outputs/step_counting \
  --apply_chat_template \
  --system_instruction "You are a helpful assistant that can assist users with mathematical reasoning." \
  --fewshot_as_multiturn \
  --gen_kwargs "criterion=entropy-diff,exit_threshold=1e-2,cache_lookup_strategy=latest-m4-compress-s16"
"""

"""
python CUDA_VISIBLE_DEVICES=1 evaluate_raven/record_steps_in_bench.py \
  --model hf \
  --model_args "pretrained=tomg-group-umd/huginn_swa_75_7_ema_0.9_merge,trust_remote_code=True,dtype=bfloat16,mean_recurrence=32" \
  --tasks gsm8k_cot
"""
