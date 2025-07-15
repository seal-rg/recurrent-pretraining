import os
import sys

# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from evaluate_raven.hf_eval_adaptive_compute import evaluate_single_task
from recpre.raven_modeling_minimal import NumStepsGenerator


if __name__ == "__main__":
    # simple example demonstrating how to use the exit evaluator
    exit_evaluator = NumStepsGenerator(lambda x: 4 if x % 2 == 0 else 8)

    task = "hellaswag"
    batch_size = 1
    max_steps = 8

    evaluate_single_task(
        task_name=task,
        batch_size=batch_size,
        num_steps=max_steps,
        num_fewshot=0,
        exit_evaluator=exit_evaluator,
        limit=100,
        output_filepath=f"{task}_results_48.json",
    )
