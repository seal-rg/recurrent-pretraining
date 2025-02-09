import torch
from transformers import AutoTokenizer
import sys
from pathlib import Path
import os

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import warnings

warnings.filterwarnings("ignore", message="The config.capture_autograd_function flag is deprecated")  # pytorch nightly
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`.*")  # our weights

from recpre.config_dynamic import RecurrentConfig

from typing import List
from dataclasses import dataclass
import time

import pickle


def load_specific_keys(checkpoint_name, keys=["model_config", "model"]):
    loaded_state = {}
    pickle_load = pickle.load

    def selective_load(*args, **kwargs):
        data = pickle_load(*args, **kwargs)
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in keys}
        return data

    try:
        # Temporarily override pickle.load
        pickle.load = selective_load
        loaded_state = torch.load(checkpoint_name, map_location="cpu")
    finally:
        # Restore original pickle.load
        pickle.load = pickle_load

    return loaded_state


@dataclass
class Message:
    role: str
    content: str


class ChatInterface:
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        debug: bool = False,
        recurrence_steps: List[int] = [5, 10, 20, 30],  # Default steps to compare
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.amp_settings = {"device_type": "cuda", "enabled": True, "dtype": torch.bfloat16}
        self.debug = debug
        self.recurrence_steps = recurrence_steps
        num_params = sum([p.numel() for p in model.parameters()])
        rec_params = sum([p.numel() for p in model.transformer.core_block.parameters()])
        static_params = num_params - rec_params
        r = model.config.mean_recurrence
        unfolded_params_mean = static_params + rec_params * r
        unfolded_params_max = static_params + rec_params * 2 * r
        print(f"Model loaded with {int(rec_params / 1e6):,}m parameters in recurrent block.")
        print(f"Will unfold to {int(unfolded_params_mean // 1e6):,}m mean parameters at test time ({r} rec).")
        print(f"Could unfold to {int(unfolded_params_max // 1e6):,}m parameters at test time ({2 * r} rec).")

    def generate_with_recurrence(self, input_ids: torch.Tensor, num_steps: int) -> str:
        """Generate response with specific recurrence steps."""
        past_tokens = input_ids
        generated_text = ""
        num_steps_pair = torch.tensor([num_steps, 0], device=self.device)

        with torch.no_grad():
            for _ in range(self.max_new_tokens):
                with torch.autocast(**self.amp_settings):
                    outputs = self.model(
                        input_ids=past_tokens[:, -self.model.config.block_size :],
                        attention_mask=None,
                        return_logits=True,
                        num_steps_pair=num_steps_pair,  # Add recurrence control
                    )
                probs = torch.softmax(outputs["logits"][:, -1, :].float() / self.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                past_tokens = torch.cat([past_tokens, next_token], dim=1)
                new_text = self.tokenizer.decode(past_tokens[0, input_ids.shape[1] :])

                if "<|end_turn|>" in new_text or "<|end_text|>" in new_text:
                    generated_text = new_text.split("<|end_turn|>")[0].split("<|end_text|>")[0]
                    break

                generated_text = new_text
                if not self.debug:
                    print("." if _ % 3 == 0 else " ", end="\r", flush=True)

        return generated_text.strip()

    def generate_response(self, messages: List[Message]) -> str:
        formatted_messages = [
            {"role": "Huginn" if m.role == "assistant" else m.role, "content": m.content.strip()} for m in messages
        ]

        chat_input = self.tokenizer.apply_chat_template(formatted_messages, tokenize=False, add_generation_prompt=True)
        input_ids = self.tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(self.device)

        if self.debug:
            print("\nDebug: Generating responses with different recurrence steps:")
            print("=" * 60)

            # Generate responses with different recurrence steps
            responses = {}
            for steps in self.recurrence_steps:
                start_time = time.time()
                response = self.generate_with_recurrence(input_ids, steps)
                end_time = time.time()
                responses[steps] = (response, end_time - start_time)

                print(f"\nRecurrence steps: {steps}")
                print(f"Generation time: {end_time - start_time:.2f}s")
                print("-" * 40)
                print(response)
                print("-" * 40)

            # Return the response with the default (middle) recurrence steps
            default_steps = 32  # trained recurrence
            return responses[default_steps][0]
        else:
            # Normal mode: just generate with default recurrence
            return self.generate_with_recurrence(input_ids, 32)


def quick_chat_interface(
    checkpoint_name: str = "step-00047360-recurrence_full_512_0.pth",  #  "/is/cluster/fast/jgeiping/recllm/outputs/magpie2/checkpoints-DDPStrategy/step-00439553-magpie_cooldown_32_8.pth",
    tokenizer_path: str = "tomg-group-umd/huginn_tokenizer_65k",
    device: torch.device = torch.device("cuda:0"),
    recurrence: int = 32,
    debug: bool = True,
    recurrence_steps: List[int] = [4, 8, 16, 20, 32, 64],
    temperature: float = 0.7,
):
    # Initialize model and tokenizer
    print("Loading tokenizer and model...")
    print(checkpoint_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert os.path.realpath(checkpoint_name)
    print(os.stat(checkpoint_name))

    state = load_specific_keys(checkpoint_name)
    config = RecurrentConfig.from_name(state["model_config"]["name"])
    config.attn_impl = "sdpa"
    config.mean_recurrence = recurrence  # type: ignore
    model = config.construct_model(objective=[], tokenizer=tokenizer, gradient_checkpointing=False)
    model.load_state_dict(
        {
            k.replace("_orig_mod._original_module.", ""): v
            for k, v in state["model"].items()
            if "forward_module" not in k
        }
    )
    model = model.to(device=device)
    model.eval()

    chat = ChatInterface(
        model, tokenizer, device, debug=debug, recurrence_steps=recurrence_steps, temperature=temperature
    )
    messages: List[Message] = []

    print("\nChat initialized! Type 'quit' to exit, 'clear' to reset chat history.")
    print("Model parameters:", f"temperature={chat.temperature}", f"max_tokens={chat.max_new_tokens}")
    if debug:
        print(f"Debug mode: ON - Will show outputs for recurrence steps: {recurrence_steps}")

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "quit":
                break
            elif user_input.lower() == "clear":
                messages = []
                print("Chat history cleared!")
                continue
            elif not user_input:
                continue

            messages.append(Message(role="user", content=user_input))
            print("\nAssistant: ", end="", flush=True)

            start_time = time.time()
            response = chat.generate_response(messages)
            end_time = time.time()

            if not debug:
                print(" " * 20, end="\r", flush=True)
                print(response)
                print(f"\n[Generated in {end_time - start_time:.2f}s]")

            messages.append(Message(role="assistant", content=response))

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue


from jsonargparse import CLI

if __name__ == "__main__":
    CLI(quick_chat_interface)
