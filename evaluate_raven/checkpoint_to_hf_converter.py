import torch
from pathlib import Path
from jsonargparse import CLI


import sys
import time

# support running without installing as a package because we're not bothering to fix the package structure
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import recpre  # noqa: F401 # import to register autoclass
from recpre.raven_modeling_minimal import RavenForCausalLM, RavenConfig, HuginnDynamicCache
from transformers import AutoTokenizer
from recpre.config_dynamic import RecurrentConfig

amp_settings = {"device_type": "cuda", "enabled": False, "dtype": torch.bfloat16}
if not amp_settings["enabled"]:
    torch.backends.cuda.enable_math_sdp(True)


def mini_generate(
    model,
    tokenizer,
    input_ids,
    num_steps=32,
    device=torch.device("cuda"),
    max_new_tokens: int = 200,
    temperature=0,
    use_cache=False,
    deterministic=True,
):
    t0 = time.time()
    num_steps = torch.tensor([num_steps, 0], device=device)
    past_tokens = input_ids
    generated_text = ""
    if use_cache:
        past_key_values = HuginnDynamicCache()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.autocast(**amp_settings):
                if use_cache:
                    input_length = past_tokens[:, past_key_values.get_seq_length() :].shape[1]
                    positions = past_key_values.get_seq_length() + torch.arange(input_length, device=device)
                    outputs = model(
                        input_ids=past_tokens[:, past_key_values.get_seq_length() :],
                        attention_mask=None,
                        num_steps=num_steps,
                        past_key_values=past_key_values,
                        position_ids=positions,
                        input_states=torch.zeros((1, input_length, model.config.n_embd), device=device)
                        if deterministic
                        else None,
                    )
                else:
                    outputs = model(
                        input_ids=past_tokens,
                        attention_mask=None,
                        num_steps=num_steps,
                        input_states=torch.zeros((1, past_tokens.shape[1], model.config.n_embd), device=device)
                        if deterministic
                        else None,
                    )
            # print(outputs.logits[:, -1, :].float())
            if temperature > 0:
                probs = torch.softmax(outputs.logits[:, -1, :].float() / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(outputs.logits[:, -1, :].float(), dim=-1)[None]

            past_tokens = torch.cat([past_tokens, next_token], dim=1)
            new_text = tokenizer.decode(past_tokens[0, input_ids.shape[1] :])

            if "<|end_turn|>" in new_text or "<|end_text|>" in new_text:
                generated_text = new_text.split("<|end_turn|>")[0].split("<|end_text|>")[0]
                break
            generated_text = new_text

    print(f"Generation in {time.time() - t0}s")
    return generated_text.strip()


def hf_implementation_test_and_upload(
    checkpoint_name: str = "step-00047360-recurrence_full_512_0.pth",
    tokenizer_path="tomg-group-umd/huginn_tokenizer_65k",
    device: torch.device = torch.device("cuda:0"),
    out_dir="outputs/eval",
    upload: bool = True,
    compare_to_pretrain_implementation=False,
    hf_gen_crosscheck=False,
) -> None:
    # print gpus available
    print(f"GPU Info: {torch.cuda.device_count()} - {torch.cuda.current_device()}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = checkpoint_name.split("/")[-1].split(".")[0]
    # Pretrain state
    print(f"Checkpoint: {checkpoint_name} -- Model name will be {model_name}")
    t0 = time.time()
    state = torch.load(checkpoint_name, map_location="cpu", weights_only=False, mmap=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    if compare_to_pretrain_implementation:
        # Old implementation:
        old_config = RecurrentConfig.from_name(state["model_config"]["name"])
        old_config.attn_impl = "sdpa"
        old_config.mean_recurrence = 32  # type: ignore
        pretrain_model = old_config.construct_model(objective=[], tokenizer=tokenizer, gradient_checkpointing=False)
        missing_keys, unexpected_keys = pretrain_model.load_state_dict(
            {
                k.replace("_orig_mod._original_module.", ""): v
                for k, v in state["model"].items()
                if "_orig_mod._original_module." in k
            }
        )
        print(missing_keys, unexpected_keys)

        pretrain_model = pretrain_model.to(device=device)
        pretrain_model.eval()

    # New implementation
    config = RavenConfig()
    with torch.device("meta"):
        model = RavenForCausalLM(config)
    missing_keys, unexpected_keys = model.load_state_dict(
        {
            k.replace("_orig_mod._original_module.", ""): v
            for k, v in state["model"].items()
            if "forward_module" not in k
        },
        assign=True,
    )
    print(missing_keys, unexpected_keys)
    print(f"Params loaded in {time.time() - t0}s.")
    model.to(device=device)  # type: ignore
    model.eval()
    print(f"Model ready on device in {time.time() - t0}s.")

    test_sentence = "The capital of Westphalia is"
    input_ids = tokenizer.encode(test_sentence, return_tensors="pt", add_special_tokens=True).to(device)[:, :-1]  # type: ignore

    if compare_to_pretrain_implementation:
        with torch.autocast(**amp_settings), torch.no_grad():
            pretrain_logits = pretrain_model(input_ids, return_logits=True)["logits"]  # type: ignore
            hf_impl_logits = model(input_ids=input_ids, use_cache=True).logits
            print((pretrain_logits - hf_impl_logits).norm(dim=-1))

    # Test cache vs no-cache generation
    generated_text = mini_generate(
        model, tokenizer, input_ids, num_steps=32, device=torch.device("cuda"), max_new_tokens=20, temperature=0
    )
    print(generated_text)
    generated_text = mini_generate(
        model,
        tokenizer,
        input_ids,
        num_steps=32,
        device=torch.device("cuda"),
        max_new_tokens=20,
        temperature=0,
        use_cache=True,
    )
    print(generated_text)
    if upload:
        model.save_pretrained("huginn_raven_test_upload")
        model.push_to_hub(repo_id=model_name, organization="tomg-group-umd", private=True)  # type: ignore
        # sean template:
        chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set start_content = '<|begin_header|>' %}{% set end_content = message['content'] | trim  + '<|end_turn|>' %}{% if loop.index0 == 0 %}{% set start_content = bos_token + start_content %}{% endif %}{% if message['role'] == 'Huginn' or message['role'] == 'assistant' %}{% set start_content = start_content + 'Huginn<|end_header|>\n\n' %}{{ start_content }}{% generation %}{{ end_content }}{% endgeneration %}{% else %}{% set start_content = start_content + message['role'] + '<|end_header|>\n\n' %}{{ start_content }}{{ end_content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% else %}{{ '<|end_text|>' }}{% endif %}"
        tokenizer.chat_template = chat_template
        tokenizer.push_to_hub(model_name, organization="tomg-group-umd", private=True)

    # Crosscheck to HF generate
    if hf_gen_crosscheck:
        from transformers import AutoModelForCausalLM

        hf_model = AutoModelForCausalLM.from_pretrained(f"tomg-group-umd/{model_name}", trust_remote_code=False)
        hf_model.to(device=device)  # type: ignore
        hf_model.eval()

        # Basic eval:
        print(model(input_ids).logits - hf_model(input_ids).logits)

        # # with cache
        with torch.autocast(**amp_settings), torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=20, use_cache=True, num_steps=32)
        print(tokenizer.decode(output_ids[0]))
        # no cache
        with torch.autocast(**amp_settings), torch.no_grad():
            output_ids = hf_model.generate(input_ids, max_new_tokens=20, use_cache=False, num_steps=32)
        print(tokenizer.decode(output_ids[0]))


if __name__ == "__main__":
    CLI(hf_implementation_test_and_upload)
