from lm_eval.api.model import TemplateLM


from typing import List, Tuple, TypedDict, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))


from recpre.config_dynamic import RecurrentConfig
from jsonargparse import CLI
from tqdm import tqdm
import json
import os

import warnings

warnings.filterwarnings("ignore", message="The config.capture_autograd_function flag is deprecated")  # pytorch nightly
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")  # our weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import datasets

datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True  # type: ignore


def set_internet_env_variables():
    return """
export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy="http://proxy.ccs.ornl.gov:3128/"
export HTTPS_PROXY="http://proxy.ccs.ornl.gov:3128/"
"""


class AmpSettings(TypedDict):
    device_type: str
    dtype: torch.dtype
    enabled: bool


class RecurrentGPTWrapper(TemplateLM):
    """Clauded wrapper for RecurrentGPT using only its forward method."""

    def __init__(
        self,
        ckpt_state_path: str,
        tokenizer_path: str,
        device=torch.device("cpu"),
        batch_size: int = 8,
        use_mixed_precision=True,
        recurrence=32,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        state = torch.load(ckpt_state_path, map_location="cpu")
        self.config = RecurrentConfig.from_name(state["model_config"]["name"])
        self.config.attn_impl = "sdpa"
        self.config.mean_recurrence = recurrence  # type: ignore
        print(self.config)
        self.model = self.config.construct_model(objective=[], tokenizer=self.tokenizer, gradient_checkpointing=False)
        self.model.load_state_dict(
            {
                k.replace("_orig_mod._original_module.", ""): v
                for k, v in state["model"].items()
                if "_orig_mod._original_module." in k
            }
        )
        self.model = self.model.to(device=device)
        self.model.eval()

        self.max_gen_toks = 512  # for efficiency

        self.device = device
        self.batch_size = batch_size
        self.max_length = self.config.block_size
        self.use_mixed_precision = use_mixed_precision
        self.amp_settings = {
            "device_type": "cuda",
            "enabled": use_mixed_precision,
            "dtype": torch.bfloat16,
        }
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(True)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id  # type: ignore

    def tok_encode(self, string: str, **kwargs) -> List[int]:
        return self.tokenizer.encode(string, **kwargs, add_special_tokens=False)

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        override_bs: Optional[int] = None,
    ) -> List[Tuple[float, bool]]:
        """
        Computes log-likelihoods for a batch of context and continuation token sequences.

        Args:
            requests: List of tuples containing (context_str, continuation_str), context_tokens, continuation_tokens
            disable_tqdm: Whether to disable the progress bar
            override_bs: Override batch size if provided

        Returns:
            List of tuples containing (log_likelihood, is_greedy) for each request
        """
        # Use provided batch size or override
        batch_size = self.batch_size
        results = []

        # Process requests in batches
        for i in tqdm(range(0, len(requests), batch_size), disable=disable_tqdm, desc="Computing loglikelihoods"):
            # for i in range(0, len(requests), batch_size):
            batch = requests[i : i + batch_size]
            batch_inps = []
            batch_cont_toks = []
            batch_inplens = []

            # Prepare inputs for each request in batch
            for _, context_enc, continuation_enc in batch:
                # Concatenate context and continuation, leaving off last token
                # Format: BOS + context + continuation[:-1]
                inp = torch.tensor(
                    ([self.tokenizer.bos_token_id] + context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                inplen = inp.shape[0]

                batch_inps.append(inp)
                batch_cont_toks.append(continuation_enc)
                batch_inplens.append(inplen)

            # Pad sequences in batch to same length
            padding_length = max(len(inp) for inp in batch_inps)
            padded_inps = []

            for inp in batch_inps:
                if len(inp) < padding_length:
                    # Right pad with pad token
                    padded_inp = torch.cat(
                        [inp, torch.full((padding_length - len(inp),), self.tokenizer.pad_token_id, device=self.device)]  # type: ignore
                    )
                    padded_inps.append(padded_inp)
                else:
                    padded_inps.append(inp)

            batch_inps = torch.stack(padded_inps)

            # Get model predictions
            with torch.no_grad(), torch.autocast(**self.amp_settings):
                logits = self.model(batch_inps, return_logits=True)["logits"]

            # Convert to log probabilities
            log_probs = F.log_softmax(logits, dim=-1)

            # Calculate result for each sample in batch
            for log_prob_seq, inplen, cont_toks in zip(log_probs, batch_inplens, batch_cont_toks):
                # use tdqm to show progress bar
                # for log_prob_seq, inplen, cont_toks in tqdm(
                #     zip(log_probs, batch_inplens, batch_cont_toks),
                #     disable=disable_tqdm,
                #     desc="Processing batch samples",
                # ):
                # Get logits for continuation tokens only
                cont_len = len(cont_toks)
                relevant_logits = log_prob_seq[inplen - cont_len : inplen]

                # Get log probs for the actual continuation tokens
                cont_token_tensors = torch.tensor(cont_toks, device=self.device)
                cont_log_probs = torch.gather(relevant_logits, 1, cont_token_tensors.unsqueeze(1)).squeeze(1)

                # Sum log probs
                total_log_prob = cont_log_probs.sum().item()

                # Check if continuation matches greedy prediction
                pred_tokens = relevant_logits.argmax(dim=1)
                is_greedy = (pred_tokens == cont_token_tensors).all().item()

                results.append((total_log_prob, is_greedy))
        print("Finished computing loglikelihoods.")
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> List[float]:
        """
        Computes rolling log-likelihoods for a list of text strings.

        Args:
            requests: List of text requests
            disable_tqdm: Whether to disable the progress bar

        Returns:
            List of log-likelihoods for each request
        """
        loglikelihoods = []

        for (string,) in tqdm(
            [req.args for req in requests], disable=disable_tqdm, desc="Computing rolling loglikelihoods"
        ):
            # Get token windows with context length of 1
            token_list = self.tok_encode(string)
            windows = []

            for i in range(1, len(token_list)):
                # Context is just the previous token
                context = token_list[i - 1 : i]
                # Continuation is the current token
                continuation = token_list[i : i + 1]
                windows.append((("", ""), context, continuation))

            # Get log-likelihood for each window
            window_lls = self._loglikelihood_tokens(windows, disable_tqdm=True)

            # Sum the log-likelihoods (discarding is_greedy)
            total_ll = sum(ll for ll, _ in window_lls)
            loglikelihoods.append(total_ll)

        return loglikelihoods

    def generate_until(self, requests, disable_tqdm: bool = False) -> List[str]:
        """
        Generates text for each request until EOS token is reached.
        Uses simple greedy decoding as we're only using this for evaluation.

        Args:
            requests: List of instances containing generation requests
            disable_tqdm: Whether to disable progress bar

        Returns:
            List of generated text strings
        """
        results = []

        for request in tqdm(requests, disable=disable_tqdm, desc="Generating responses"):
            context, gen_kwargs = request.args

            # Extract stop sequences if any
            kwargs = gen_kwargs.copy() if isinstance(gen_kwargs, dict) else {}
            until = kwargs.pop("until", None)
            stop_sequences = []
            if until:
                if isinstance(until, str):
                    stop_sequences = [until]
                elif isinstance(until, list):
                    stop_sequences = until

            # Get max tokens to generate
            max_new_tokens = kwargs.pop("max_gen_toks", self.max_gen_toks)

            # Encode context
            context_tokens = self.tok_encode(context)
            context_tensor = torch.tensor(context_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            # Generate tokens
            with torch.no_grad():
                output_ids = []
                current_input = context_tensor

                for _ in range(max_new_tokens):
                    # Get model predictions
                    with torch.no_grad(), torch.autocast(**self.amp_settings):
                        logits = self.model(current_input, return_logits=True)["logits"]
                    next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                    next_token_id = next_token.item()
                    output_ids.append(next_token_id)

                    # Check for EOS token first
                    if next_token_id == self.eot_token_id:
                        break

                    # Then check other stop sequences if any
                    if stop_sequences:
                        generated_text = self.tok_decode(output_ids)
                        if any(seq in generated_text for seq in stop_sequences):
                            break

                    # Update input for next iteration
                    current_input = torch.cat([current_input, next_token], dim=-1)

                    # Truncate if exceeding max length
                    if current_input.size(1) > self.max_length:
                        current_input = current_input[:, -self.max_length :]

                # Decode the generated tokens
                generated_text = self.tok_decode(output_ids)

                # Apply stop sequence trimming if needed
                if stop_sequences:
                    for stop_seq in stop_sequences:
                        if stop_seq in generated_text:
                            generated_text = generated_text.split(stop_seq)[0]
                            break

                results.append(generated_text)

        return results


def prepare_results(results, save_filepath, print_results=True):
    from lm_eval.utils import make_table

    if print_results:
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

    json_result = json.dumps(results, indent=2, ensure_ascii=False, default=str)
    save_filepath.open("w", encoding="utf-8").write(json_result)
    # return make_table(results, "groups")


def quick_eval_check(
    checkpoint_name="/is/cluster/fast/jgeiping/recllm/outputs/magpie2/checkpoints-DDPStrategy/step-00439553-magpie_cooldown_32_8.pth",
    tokenizer_path="/lustre/orion/csc569/scratch/njain17/new_workspace/holder/lit-gpt-dev/jonas_models/hf_model_12k_pretrained",
    device=torch.device("cuda"),
    out_dir="outputs/eval",
    # tasks="arc_easy,hellaswag",
    # tasks="arc_easy,hellaswag,triviaqa,tinyMMLU",
    # tasks="arc_easy,hellaswag,triviaqa,tinyMMLU,piqa,openbookqa",
    # tasks="arc_easy,hellaswag,triviaqa,tinyMMLU,piqa,openbookqa",
    tasks="arc_easy,tinyHellaswag,tinyMMLU,piqa,openbookqa",
    # gsm8k_cot,mathqa,bbh_fewshot_logical_deduction_five_objects,mmlu_abstract_algebra,agieval_math
    num_fewshot=0,
    batch_size=16,
    limit=None,
    seed=233,
    recurrence=32,
):
    # print gpus available
    print(f"Available GPUs: {torch.cuda.device_count()}")
    # print which gpus are available
    print(f"Available GPUs: {torch.cuda.current_device()}")
    if tasks is None:
        from lm_eval.tasks import TaskManager

        taskm = TaskManager()
        print("\n".join(taskm.task_index.keys()))
        print(
            "\n\nTo evaluate multiple tasks, you can chain the task names "
            "listed above via a comma-separated list."
            "\nFor example: `--tasks 'hellaswag,truthfulqa_mc2,mmlu'`. "
            "\nTo search for a specific task, use `recpre evaluate list | grep task_name`."
        )
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = checkpoint_name.split("/")[-1].split(".")[0]
    base_dir = out_dir / model_name
    save_filepath = out_dir / Path(f"results_{tasks}_{model_name}_{recurrence}.json")

    from lm_eval import evaluator

    print(f"Running evaluation on tasks: {tasks}")
    print(f"Checkpoint: {checkpoint_name}")
    model = RecurrentGPTWrapper(
        checkpoint_name, tokenizer_path, device=device, batch_size=batch_size, recurrence=recurrence
    )

    results = evaluator.simple_evaluate(
        model=model,
        tasks=tasks.split(","),  # type: ignore
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=str(device),
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    prepare_results(results, save_filepath)

    # log results to wandb
    result_dict = {
        key: round(value.get("acc_norm,none", value.get("acc,none")), 4)
        for key, value in results["results"].items()  # type: ignore
    }
    import wandb

    wandb.init(
        entity="tomg-group-umd", project="recurrence_lm_eval_test", name=f"{model_name}_{recurrence}rec", dir=out_dir
    )
    wandb.log({"recurrence": recurrence, "base_dir": str(base_dir)})
    steps = int(model_name.split("-")[-2])
    wandb.log(result_dict, step=steps)
    print(f"result_dict: {result_dict}")


if __name__ == "__main__":
    CLI(quick_eval_check)
