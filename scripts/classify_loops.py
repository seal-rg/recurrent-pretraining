import os
import sys
import torch
import torch.nn as nn
from typing import Iterator, Optional
import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

import random
from torch.utils.data.dataloader import default_collate


# Add the parent directory to the Python path to make recpre importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recpre.raven_modeling_minimal import RavenConfig, RavenForCausalLM


@torch.no_grad()
def capture_forward_latents(
    self: RavenForCausalLM,
    input_ids: torch.Tensor,
    num_steps: int,
    init_scale: float = 1.0,
    **kwargs,
) -> list[torch.Tensor]:
    latents = []

    freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
    input_embeds = self.transformer.wte(input_ids)  # type: ignore # types broken in 2.6+

    if self.emb_scale != 1:
        input_embeds = input_embeds * self.emb_scale  # type: ignore

    block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)  # count in tensors for compile
    # Non-recurrent prelude
    for block in self.transformer.prelude:  # type: ignore # types broken in 2.6+
        block_idx += 1
        input_embeds = block(input_embeds, freqs_cis, block_idx)
    current_latents = self.initialize_state(input_embeds, scale=init_scale)
    latents.append(current_latents.clone().detach())
    # Main recurrence
    for compute_step in range(num_steps):
        current_latents, block_idx, _ = self.iterate_one_step(
            input_embeds,
            current_latents,
            block_idx=block_idx,
            current_step=compute_step,
        )
        latents.append(current_latents.clone().detach())

    return latents


class GatedMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: Optional[int] = None, output_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        if output_dim is None:
            output_dim = input_dim
        self.fc = nn.Linear(input_dim, hidden_dim * 2)
        self.nonlin = nn.SiLU()
        self.proj = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor):
        x_fc_1, x_fc_2 = self.fc(x).chunk(2, dim=-1)
        x = self.nonlin(x_fc_1) * x_fc_2
        return self.proj(x)


class LoopClassifier(nn.Module):
    def __init__(self, config: RavenConfig, max_loops: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.mlp1 = GatedMLP(config.n_embd)
        self.out = nn.Linear(config.n_embd, max_loops)

        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor):
        x = self.norm1(x + self.mlp1(x))
        return self.out(x).float()


def compute_classify_loops_loss(
    model: RavenForCausalLM,
    loop_predictor: LoopClassifier,
    data_iter: Iterator[list[list[int]]],
    min_steps: int,
    max_steps: int,
    loops_per_compare: int = 1,
):
    with torch.no_grad():
        x = torch.tensor(next(data_iter), dtype=torch.int64, device="cuda")
        latent_list: list[torch.Tensor] = capture_forward_latents(model, x, max_steps)
        # latents are of shape (num_steps, batch_size, seq_len, n_embd)
        latents = torch.stack(latent_list[min_steps::loops_per_compare], dim=0).to(dtype=loop_predictor.out.weight.dtype)
    # preds are of shape (num_steps, batch_size, seq_len, max_loops)
    preds = loop_predictor(latents)
    targets = torch.zeros_like(preds)
    for i in range(min_steps, max_steps + 1, loops_per_compare):
        offset_idx = i - min_steps
        targets[offset_idx // loops_per_compare, :, :, offset_idx // loops_per_compare] = 1.0
    loss = torch.nn.functional.cross_entropy(preds.view(-1, preds.shape[-1]), targets.view(-1, targets.shape[-1]))
    return loss


def train_loop_predictor(
    model: RavenForCausalLM,
    loop_predictor: LoopClassifier,
    dataloader: DataLoader,
    num_epochs: int,
    optimizer: torch.optim.Optimizer,
    min_steps: int = 0,
    max_steps: int = 32,
    loops_per_compare: int = 1,
    val_batchs: int = 10,
    save_prefix: str = "",
):
    data_iter = iter(dataloader)
    x_val = []
    for i in range(val_batchs):
        # NOTE: this is technically not a validation set since we don't exclude the validation set from the training set
        #   But due to how big RedPajama is, we don't do enough iterations to see the same data twice, so it's fine
        x = torch.tensor(next(data_iter), dtype=torch.int64, device="cuda")
        x_val.append(x)

    loop_predictor.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = compute_classify_loops_loss(model, loop_predictor, data_iter, min_steps, max_steps, loops_per_compare=loops_per_compare)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 9:
            loop_predictor.eval()
            with torch.no_grad():
                val_loss = 0
                for x in x_val:
                    loss = compute_classify_loops_loss(model, loop_predictor, data_iter, min_steps, max_steps, loops_per_compare=loops_per_compare)
                    val_loss += loss.item()
                val_loss /= len(x_val)
                logging.info(f"{save_prefix}: Epoch {epoch}, Val Loss {val_loss}")
            loop_predictor.train()
        if epoch % 500 == 499:
            # Save the model
            torch.save(loop_predictor.state_dict(), f"{save_prefix}loop_predictor_{epoch}.pt")

    return loop_predictor, val_loss


def get_redpajama_dataloader(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, batch_size: int, seq_len: int):
    # Load RedPajama dataset (default mix)
    redpajama_stream = load_dataset("togethercomputer/RedPajama-Data-1T", "default", split="train", streaming=True)

    def map_fn(tokens):
        tokens_length = len(tokens)
        if tokens_length < seq_len:
            tokens = tokens + [tokenizer.pad_token_id] * (seq_len - tokens_length)
        else:
            random_indice = random.randint(0, tokens_length - seq_len)
            tokens = tokens[random_indice:random_indice+seq_len]
        return tokens

    def collate_fn(batch):
        examples = [d["text"] for d in batch]
        tokens = tokenizer(examples)["input_ids"]
        tokens = [map_fn(t) for t in tokens] # type: ignore
        return tokens

    return DataLoader(
        redpajama_stream, # type: ignore
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=1,
        persistent_workers=False,
        prefetch_factor=2,
    )


def train_all_loop_classifier():
    """
    Trains a classifier for latents from r=0-32.
    """
    torch.manual_seed(42)

    # Set up logging
    log_file = "training_loop_predictor.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.to("cuda", dtype=torch.bfloat16) # type: ignore

    batch_size = 4
    seq_len = 256
    dataloader = get_redpajama_dataloader(tokenizer, batch_size, seq_len)

    max_epochs = 5000
    min_steps = 0
    max_steps = 32
    loop_predictor = LoopClassifier(model.config, max_steps - min_steps + 1)
    loop_predictor.to(model.device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(loop_predictor.parameters(), lr=0.001)

    train_loop_predictor(model, loop_predictor, dataloader, max_epochs, optimizer, min_steps=min_steps, max_steps=max_steps)


def train_binary_loop_classification_sweep(loops_per_compare: int = 1):
    """
    Train a lot of binary loop classifier of i-th loop vs i+<loops_per_compare>-th loop.
    The validation losses of each classifier are printed
    """
    torch.manual_seed(42)

    # Set up logging
    log_file = "training_loop_predictor.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    model: RavenForCausalLM = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.to("cuda", dtype=torch.bfloat16) # type: ignore

    batch_size = 4
    seq_len = 256
    max_epochs = 500

    val_losses = []
    for step in range(0, 32, loops_per_compare):
        min_steps = step
        max_steps = step + loops_per_compare
        dataloader = get_redpajama_dataloader(tokenizer, batch_size, seq_len)
        loop_predictor = LoopClassifier(model.config, 2)
        loop_predictor.to(model.device, dtype=torch.bfloat16)

        optimizer = torch.optim.AdamW(loop_predictor.parameters(), lr=0.001)

        _, val_loss = train_loop_predictor(model, loop_predictor, dataloader, max_epochs, optimizer, min_steps=min_steps, max_steps=max_steps, loops_per_compare=loops_per_compare, save_prefix=f"loops_{min_steps}vs{max_steps}")
        val_losses.append((min_steps, max_steps, val_loss))

    logging.info(f"Val Losses: {val_losses}")
    print(val_losses)


if __name__ == "__main__":
    # train_all_loop_classifier()
    train_binary_loop_classification_sweep(loops_per_compare=1)
