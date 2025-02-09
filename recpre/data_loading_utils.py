import torch

from typing import Optional
from torch.utils.data._utils.collate import collate_tensor_fn
from .tokenizer import Tokenizer


def pass_text(row, tokenizer, add_bos, add_eos):
    input_string = row["text"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def concat_input_target(row, tokenizer, add_bos, add_eos):
    input_string = row["input"] + row["target"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def condition_input_supervise_target(row, tokenizer, add_bos, add_eos):
    input_string = row["input"]
    joint_string = row["input"] + row["target"]
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=False)
    joint_tokens = tokenizer.encode(joint_string, bos=add_bos, eos=add_eos)
    label_tokens = joint_tokens.clone()
    # mask the locations of the input tokens in the joint tokens
    label_tokens[0 : len(input_tokens)] = tokenizer.pad_id
    input_tokens = joint_tokens
    return (input_tokens, label_tokens)


def apply_chat_template_supervise_all(row, tokenizer, add_bos, add_eos):
    assert len(row["data_signature"]["keys"]) == 1, (
        "Ambiguous row format for chat template call. data signature should spec the single intended key."
    )
    key = row["data_signature"]["keys"][0]
    input_string = tokenizer.processor.apply_chat_template(row[key], tokenize=False)
    input_tokens = tokenizer.encode(input_string, bos=add_bos, eos=add_eos)
    label_tokens = input_tokens.clone()
    return (input_tokens, label_tokens)


def apply_chat_template_supervise_assistant(row, tokenizer, add_bos, add_eos):
    # This is temporary whilst we fix the chat template
    tokenizer.processor.chat_template = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set start_content = '<|begin_header|>' %}{% set end_content = message['content'] | trim  + '<|end_turn|>' %}{% if loop.index0 == 0 %}{% set start_content = bos_token + start_content %}{% endif %}{% if message['role'] == 'Huginn' or message['role'] == 'assistant' %}{% set start_content = start_content + 'Huginn<|end_header|>\n\n' %}{{ start_content }}{% generation %}{{ end_content }}{% endgeneration %}{% else %}{% set start_content = start_content + message['role'] + '<|end_header|>\n\n' %}{{ start_content }}{{ end_content }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% else %}{{ '<|end_text|>' }}{% endif %}"""

    assert len(row["data_signature"]["keys"]) == 1, (
        "Ambiguous row format for chat template call. data signature should spec the single intended key."
    )
    key = row["data_signature"]["keys"][0]

    assert isinstance(row[key], list), "is not in chat format"
    tokenized_string = tokenizer.processor.apply_chat_template(
        row[key],
        tokenize=True,
        add_generation_prompt=False,
        return_assistant_tokens_mask=True,
        return_dict=True,
        return_tensors="pt",
    )
    print(tokenizer.processor.decode(tokenized_string["input_ids"][0]))
    print("+" * 70)
    labels = torch.tensor(tokenized_string["assistant_masks"]) * tokenized_string["input_ids"]
    labels[labels == 0] = tokenizer.pad_id
    print(labels)
    print(tokenizer.processor.decode(labels[0]))
    exit()
    return (tokenized_string["input_ids"], labels)


format_fn_registry = {
    "pass_text": pass_text,
    "concat_input_target": concat_input_target,
    "condition_input_supervise_target": condition_input_supervise_target,
    "apply_chat_template_supervise_all": apply_chat_template_supervise_all,
    "apply_chat_template_supervise_assistant": apply_chat_template_supervise_assistant,
}


def apply_formatting(row, tokenizer, add_bos, add_eos):
    # pkds, single tensor
    if isinstance(row, torch.Tensor):
        return row, row.clone()
    # pkds, tuple of tensors
    if isinstance(row, tuple):
        raise NotImplementedError("Tuple format not supported, but direct tensor pairs planned.")
        return row[0], row[1]
    # hfds, dict with format_fn from data signature
    if isinstance(row, dict):
        # we can locally override the add_bos or add_eos args if they exist in the row's data_signature
        if row["data_signature"].get("add_bos") is not None:
            add_bos = row["data_signature"]["add_bos"]
        if row["data_signature"].get("add_eos") is not None:
            add_eos = row["data_signature"]["add_eos"]

        return format_fn_registry[row["data_signature"]["format_fn"]](row, tokenizer, add_bos, add_eos)
    raise ValueError("Row format not recognized.")


def shift_inputs_and_labels(inputs_batch: torch.Tensor, labels_batch: torch.Tensor, tokenizer: Tokenizer):
    seq_len = inputs_batch.shape[1]

    input_ids = inputs_batch[:, 0 : (seq_len - 1)].contiguous().long()
    label_ids = labels_batch[:, 1:(seq_len)].contiguous().long()

    # for the input we need to replace any pad ids with the eos token
    # knowing that they're trailing so they wont contrib to activations
    # but that they do need to be valid indices in the model's embedding layer
    if tokenizer.eos_id is not None:
        input_ids[input_ids == tokenizer.pad_id] = tokenizer.eos_id  # type: ignore
    # Note that we are _not_ doing this operation for the labels,
    # since this is where we actually need the pad tokens to be present for loss to ignore them.

    return input_ids, label_ids


def generic_collate_fn(
    batch,
    tokenizer: Tokenizer,
    block_size: Optional[int] = None,
    pad_to_block_size: bool = False,
    add_bos=True,
    add_eos=True,
    collate_checks_enabled=True,
    all_block_size_tensors=False,
):
    metadata = [None] * len(batch)
    for i, row in enumerate(batch):
        if isinstance(row, dict) and "data_id" in row:
            metadata[i] = row["data_id"]

    # If we are only dealing with tensors that we _know_ are the same size,
    # we can just use the default collate_tensor_fn.
    # this is theoretically the fastest codepath.
    # for a bleeding edge pretraining run, this flag should be set to True, all data should be pkds
    # and we do minimal to no processing on the fly.
    if all_block_size_tensors:
        inputs_batch = collate_tensor_fn(batch)
        labels_batch = inputs_batch.clone()
        input_ids, label_ids = shift_inputs_and_labels(inputs_batch, labels_batch, tokenizer)
        return input_ids, label_ids, metadata
    else:
        assert block_size is not None

    # This is O(bsz) but it's a more readable error message than the later failure would be.
    if collate_checks_enabled:
        assert isinstance(batch, list), "Batch must be a list."
        type_list = [type(x) for x in batch]
        allowed_types = [dict, torch.Tensor]
        types_found = set(type_list)
        assert types_found.issubset(allowed_types), "Batch must contain only expected types."

        if dict in types_found:
            assert tokenizer is not None, "If batch contains dicts, tokenizer must be provided."
            assert tokenizer.pad_id is not None, "Tokenizer must have pad token id since we are dynamically padding."

    # this takes in a heterogeneous list of rows and returns a batch of tensor pairs.
    batch = [apply_formatting(row, tokenizer, add_bos, add_eos) for row in batch]

    # We operate under the assumption that all rows now have a pair of tensors as their elements.
    # In both cases we'll just declare two tensors bsz x block_size
    # and copy all the input and label tokens into them.
    # but we can unify this logic with pad to longest by setting a local_block_size
    if pad_to_block_size:
        local_block_size = block_size
    else:
        all_lengths = [len(x) for row in batch for x in row]
        # min against block size since the max realized could be longer than block size.
        local_block_size = min(max(all_lengths), block_size)

    # # Impl 1: list comp row wise pad, then torch collate fn. (closer to original implementation)
    # # Using torch tensor collation is clever about writing to shm between the data and main process.
    # # But idk if this is actually faster in our setting...
    # inputs_batch = [
    #     torch.tensor(x[0][:local_block_size].tolist() + [tokenizer.pad_id] * (local_block_size - len(x[0])))
    #     for x in batch
    # ]
    # labels_batch = [
    #     torch.tensor(x[1][:local_block_size].tolist() + [tokenizer.pad_id] * (local_block_size - len(x[1])))
    #     for x in batch
    # ]
    # inputs_batch = collate_tensor_fn(inputs_batch)
    # labels_batch = collate_tensor_fn(labels_batch)

    # Impl 2: Full tensor copy version. Simpler to read, and on initial interactive tests, equivalently fast/slow.
    inputs_batch = torch.full((len(batch), local_block_size), tokenizer.pad_id or 0, dtype=torch.int)  # type: ignore
    labels_batch = torch.full((len(batch), local_block_size), tokenizer.pad_id or 0, dtype=torch.int)  # type: ignore
    for i, (input_tokens, label_tokens) in enumerate(batch):
        inputs_batch[i, : len(input_tokens)] = input_tokens[
            :local_block_size
        ]  # this ensures we don't write past the block size
        labels_batch[i, : len(label_tokens)] = label_tokens[:local_block_size]

    # Now all rows are tensors of the same, valid length, <= block_size.

    # We need to check whether the entire batch consists of padding tokens
    if torch.all(labels_batch == tokenizer.eos_id) or torch.all(labels_batch == tokenizer.pad_id):
        # if so, we raise a StopIteration to signal the exhaustion of all data sources since
        # no real tokens are present in the batch.
        raise StopIteration("All tokens in batch are padding tokens.")

    input_ids, label_ids = shift_inputs_and_labels(inputs_batch, labels_batch, tokenizer)
    return input_ids, label_ids, metadata
