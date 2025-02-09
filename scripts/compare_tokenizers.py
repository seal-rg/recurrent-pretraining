import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer

# Constants
DOWNLOADED_DATASETS_PATH = "outputs/test_download_folder"
CACHE_DIR = "outputs/test_cache"
FINAL_LOCATION = "outputs/test_processed_dataset"
LIMITER = None  # Number of rows to take from each dataset in streaming mode
os.environ["HF_DATASETS_CACHE"] = os.environ["HF_HOME"] = CACHE_DIR


# Configuration
NUM_PROC = 32
HAVE_ENOUGH_RAM = True
from tabulate import tabulate
import re
import string

import huggingface_hub

# somehow required otherwise the token is not always read correctly:
huggingface_hub.login(token=os.environ.get("MANUALLY_SET_HF_TOKEN", None), add_to_git_credential=True)


def generate_colored_summary2(labels, tokens_list, tokenizers, sequence_comparisons):
    # Define ANSI escape codes for colors (one per tokenizer)
    COLORS = ["\033[91m", "\033[92m", "\033[94m", "\033[95m", "\033[96m", "\033[93m"]
    RESET = "\033[0m"

    # Define maximum width for columns
    sequence_width = 80
    token_diff_width = 5
    token_width = 80

    # Generate detailed summary with decoded tokens
    summary = f"Tokenization Comparison: {' vs '.join(labels)}\n"

    # Add token counts
    for label, tokens in zip(labels, tokens_list):
        summary += f"Total tokens: {label} - {len(tokens)}, "
        summary += f"Unique tokens: {label} - {len(set(tokens))}\n"
    summary += "\n"

    summary += "### Top differences in tokenization:\n\n"

    # Header with all tokenizer columns
    summary += f"{'Sequence'.ljust(sequence_width)} | {'Diff'.ljust(token_diff_width)} |"
    for label in labels:
        summary += f" {label + ' Tokens'.ljust(token_width)} |"
    summary += "\n" + "-" * (sequence_width + token_diff_width + (token_width + 3) * len(labels)) + "\n"

    sequences_seen = []
    for sequence_data in sequence_comparisons:
        sequence = sequence_data[0]
        all_tokens = sequence_data[1:]

        if sequence not in sequences_seen:
            sequences_seen += [sequence]

            # Decode all tokens
            decoded_tokens = []
            for tokenizer, tokens in zip(tokenizers, all_tokens):
                decoded = [tokenizer.decode([t]) for t in tokens]
                decoded_tokens.append(decoded)

            # Calculate max difference in token counts
            token_counts = [len(tokens) for tokens in all_tokens]
            max_diff = max(token_counts) - min(token_counts)

            if max_diff > 1:
                # Generate colored versions for each tokenizer
                colored_tokens = []
                for i, curr_decoded in enumerate(decoded_tokens):
                    # Get all other tokenizers' decoded tokens
                    other_decoded = []
                    for j, other_tokens in enumerate(decoded_tokens):
                        if j != i:
                            other_decoded.extend(other_tokens)

                    # Colorize unique tokens
                    colored = "▁".join(
                        [f"{COLORS[i]}{t}{RESET}" if t not in other_decoded else t for t in curr_decoded]
                    )

                    # Calculate clean length and pad
                    clean_length = len("▁".join([t if t not in other_decoded else t for t in curr_decoded]))
                    colored = colored + " " * max(0, token_width - clean_length)
                    colored_tokens.append(colored)

                # Add to summary
                summary += (
                    f"{sequence[:sequence_width].ljust(sequence_width)} | "
                    f"{str(max_diff).ljust(token_diff_width)} | "
                )
                summary += " | ".join(colored_tokens)
                summary += " \n"

    return summary


def detailed_tokenization_comparison2(text: str, tokenizers_and_labels: list) -> str:
    """
    Provides a detailed comparison of tokenization results for specific sequences of interest.

    Args:
        text: The input text to analyze
        tokenizers_and_labels: List of (tokenizer, label) tuples
    """
    tokenizers, labels = zip(*tokenizers_and_labels)

    # Get tokens for full text
    all_tokens = []
    for tokenizer in tokenizers:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.append(tokens)

    # Get interesting sequences (same as before)
    interesting_sequences = [t[:40] for t in text.split("\n")]

    # Analyze how each tokenizer handles these sequences
    sequence_comparisons = []
    for sequence in interesting_sequences:
        comparison = [sequence]
        for tokenizer in tokenizers:
            tokens = tokenizer.encode(sequence, add_special_tokens=False)
            comparison.append(tokens)
        sequence_comparisons.append(comparison)

    # Sort by maximum token count difference
    sequence_comparisons.sort(
        key=lambda x: max(len(tok) for tok in x[1:]) - min(len(tok) for tok in x[1:]), reverse=True
    )

    # Generate summary
    summary = generate_colored_summary2(labels, all_tokens, tokenizers, sequence_comparisons)

    return summary


def categorize_token(token: str) -> str:
    """Categorizes a token based on its content."""
    if token.isalpha():
        return "word" if len(token) > 1 else "letter"
    elif token.isnumeric():
        return "number"
    elif token.isspace():
        return "whitespace"
    elif token in ".,!?;:":
        return "punctuation"
    elif token.startswith("\\"):
        return "latex_command"
    else:
        return "other"


def analyze_tokenization(tokens: List[int], tokenizer):
    """Analyzes the tokenization process, returning a summary of token categories and statistics."""
    decoded_tokens = [tokenizer.decode([t]) for t in tokens]
    categories = {}

    for token in decoded_tokens:
        category = categorize_token(token)
        categories.setdefault(category, []).append(token)

    category_counts = {category: len(tokens) for category, tokens in categories.items()}
    avg_token_length = sum(len(token) for token in decoded_tokens) / len(decoded_tokens) if decoded_tokens else 0

    return {
        "total_tokens": len(tokens),
        "categories": category_counts,
        "category_examples": categories,
        "avg_token_length": avg_token_length,
        "long_tokens": [token for token in decoded_tokens if len(token) > 3],
    }


def find_distinctive_examples(
    examples1: List[str], examples2: List[str], num_examples: int = 5
) -> Tuple[List[str], List[str]]:
    """Finds distinctive examples between two lists of tokens."""
    set1, set2 = set(examples1), set(examples2)
    distinctive1 = list(set1 - set2)
    distinctive2 = list(set2 - set1)

    common_examples = list(set1 & set2)
    if len(distinctive1) < num_examples:
        distinctive1 += common_examples[: num_examples - len(distinctive1)]
    if len(distinctive2) < num_examples:
        distinctive2 += common_examples[: num_examples - len(distinctive2)]

    return distinctive1[:num_examples], distinctive2[:num_examples]


def summarize_comparison(comparison_results: Dict[str, Dict]) -> str:
    """Generates a summary of the comparison between two tokenizers."""
    label1, label2 = list(comparison_results.keys())
    analysis1, analysis2 = comparison_results[label1], comparison_results[label2]

    summary = (
        f"{label1} produced {analysis1['total_tokens']} tokens, "
        f"{label2} produced {analysis2['total_tokens']} tokens.\n\n"
    )

    summary += "Token category comparison:\n"
    all_categories = set(analysis1["categories"].keys()) | set(analysis2["categories"].keys())
    for category in all_categories:
        count1 = analysis1["categories"].get(category, 0)
        count2 = analysis2["categories"].get(category, 0)
        examples1 = analysis1["category_examples"].get(category, [])
        examples2 = analysis2["category_examples"].get(category, [])
        distinctive1, distinctive2 = find_distinctive_examples(examples1, examples2)

        summary += f"  {category}: {label1} - {count1}, {label2} - {count2}\n"
        summary += f"    {label1} distinctive examples: {', '.join(repr(t) for t in distinctive1)}\n"
        summary += f"    {label2} distinctive examples: {', '.join(repr(t) for t in distinctive2)}\n\n"

    summary += (
        f"\nAverage token length: {label1} - {analysis1['avg_token_length']:.2f}, "
        f"{label2} - {analysis2['avg_token_length']:.2f}\n"
    )

    summary += "\nLong tokens (>3 characters) sample:\n"
    long_tokens1, long_tokens2 = find_distinctive_examples(analysis1["long_tokens"], analysis2["long_tokens"], 10)
    summary += f"  {label1}: {', '.join(long_tokens1)}\n"
    summary += f"  {label2}: {', '.join(long_tokens2)}\n"

    return summary


def identify_interesting_sequences(text: str) -> List[str]:
    """Identifies interesting sequences such as words, punctuation, and special characters."""
    words = text.split()
    punctuation = [char for char in text if char in ".,!?;:"]
    special_chars = [char for char in text if not char.isalnum() and char not in ".,!?;: "]
    return words + punctuation + special_chars


def encode_text(tokenizer, text: str) -> List[int]:
    """Encodes text using the specified tokenizer and returns the token IDs."""
    result = tokenizer.encode(text, add_special_tokens=False)
    return result if isinstance(result, list) else result["input_ids"]


def decode_tokens(tokenizer, token_ids: List[int]) -> List[str]:
    """Decodes a list of token IDs into their corresponding strings."""
    return [tokenizer.decode([tid]) for tid in token_ids]


def tokenize_sequence(tokenizer, sequence: str) -> List[str]:
    """Tokenizes a sequence and returns the decoded tokens."""
    token_ids = encode_text(tokenizer, sequence)
    return decode_tokens(tokenizer, token_ids)


def compare_tokenizations(text: str, tokenizer1, tokenizer2, label1: str, label2: str, num_sequences: int = 20) -> str:
    """
    Compares the tokenizations of two tokenizers on the same text and returns a formatted report.
    """
    # Overall statistics
    tokens1 = encode_text(tokenizer1, text)
    tokens2 = encode_text(tokenizer2, text)

    unique_tokens1 = set(decode_tokens(tokenizer1, tokens1))
    unique_tokens2 = set(decode_tokens(tokenizer2, tokens2))

    report = []
    report.append(f"## Tokenization Comparison: {label1} vs {label2}\n")
    report.append("### Overall Statistics\n")
    report.append(
        tabulate(
            [
                ["Total Tokens", len(tokens1), len(tokens2)],
                ["Unique Tokens", len(unique_tokens1), len(unique_tokens2)],
                [
                    "Average Token Length",
                    sum(len(token) for token in unique_tokens1) / len(unique_tokens1),
                    sum(len(token) for token in unique_tokens2) / len(unique_tokens2),
                ],
            ],
            headers=["Metric", label1, label2],
            tablefmt="github",
        )
    )

    # Identify sequences with significant differences
    sequences = extract_interesting_sequences(text)
    sequence_diffs = []

    for seq in sequences:
        tokens_seq1 = tokenize_sequence(tokenizer1, seq)
        tokens_seq2 = tokenize_sequence(tokenizer2, seq)
        diff = abs(len(tokens_seq1) - len(tokens_seq2))
        if diff > 0:
            sequence_diffs.append(
                {
                    "sequence": seq,
                    "tokens1": tokens_seq1,
                    "tokens2": tokens_seq2,
                    "token_count1": len(tokens_seq1),
                    "token_count2": len(tokens_seq2),
                    "difference": diff,
                }
            )

    # Sort sequences by the difference in token counts
    sequence_diffs.sort(key=lambda x: x["difference"], reverse=True)
    top_diffs = sequence_diffs[:num_sequences]

    # Detailed comparison table
    report.append("\n### Detailed Sequence Comparison\n")
    table_data = []
    for diff in top_diffs:
        table_data.append(
            [
                diff["sequence"],
                diff["token_count1"],
                diff["token_count2"],
                diff["difference"],
                diff["tokens1"],
                diff["tokens2"],
            ]
        )

    report.append(
        tabulate(
            table_data,
            headers=[
                "Sequence",
                f"{label1} Token Count",
                f"{label2} Token Count",
                "Difference",
                f"{label1} Tokens",
                f"{label2} Tokens",
            ],
            tablefmt="github",
        )
    )

    return "\n".join(report)


def extract_interesting_sequences(text: str) -> List[str]:
    """
    Extracts sequences from text that are likely to show differences in tokenization.
    This includes:
    - URLs
    - Emails
    - Dates
    - Numbers
    - Code snippets
    - Special characters
    - Long words
    """
    patterns = [
        r"https?://\S+",
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,3}(,\d{3})*(\.\d+)?\b",
        r"`[^`]+`",
        r"\$\$[^$]+\$\$",
        r"\\[a-zA-Z]+",
        r"[^\s]{15,}",  # long words or strings without spaces
    ]
    sequences = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        sequences.update(matches)

    # Additionally, select random sentences or phrases
    sentences = re.split(r"(?<=[.!?]) +", text)
    sequences.update(sentences[:10])  # Take first 10 sentences

    return list(sequences)


def strip_ansi_escape_codes(text):
    """Remove ANSI escape codes from text to get its clean length."""
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def generate_colored_summary(label1, label2, tokens1, tokens2, tokenizer1, tokenizer2, sequence_comparisons):
    # Define ANSI escape codes for colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    # Define maximum width for columns
    sequence_width = 80
    token_diff_width = 5
    token1_width = 80
    token2_width = 80

    # Generate detailed summary with decoded tokens
    summary = f"Tokenization Comparison: {label1} vs {label2}\n"
    summary += f"Total tokens: {label1} - {len(tokens1)}, {label2} - {len(tokens2)}\n"
    summary += f"Unique tokens: {label1} - {len(set(tokens1))}, {label2} - {len(set(tokens2))}\n\n"

    summary += "### Top differences in tokenization:\n\n"
    summary += f"{'Sequence'.ljust(sequence_width)} | {'Token Count Difference'.ljust(token_diff_width)} | {label1 + ' Tokens'.ljust(token1_width)} | {label2 + ' Tokens'}\n"
    summary += "-" * (sequence_width + token_diff_width + token1_width + token2_width + 6) + "\n"

    sequences_seen = []
    for sequence, tok1, tok2 in sequence_comparisons:
        if sequence not in sequences_seen:
            sequences_seen += [sequence]
            decoded_tok1 = [tokenizer1.decode([t]) for t in tok1]
            decoded_tok2 = [tokenizer2.decode([t]) for t in tok2]
            difference = len(tok1) - len(tok2)

            if abs(difference) > 0:
                # Colorize token parts
                colored_tok1 = "▁".join([f"{RED}{t}{RESET}" if t not in decoded_tok2 else t for t in decoded_tok1])
                colored_tok2 = "▁".join([f"{GREEN}{t}{RESET}" if t not in decoded_tok1 else t for t in decoded_tok2])

                # Calculate lengths without ANSI codes
                clean_tok1_length = len("▁".join([t if t not in decoded_tok2 else t for t in decoded_tok1]))
                clean_tok2_length = len("▁".join([t if t not in decoded_tok1 else t for t in decoded_tok2]))

                # Adjust lengths to fit within the column width
                clean_tok1 = colored_tok1 + " " * max(0, token1_width - clean_tok1_length)
                clean_tok2 = colored_tok2 + " " * max(0, token2_width - clean_tok2_length)

                # Add to summary
                summary += (
                    f"{sequence[:sequence_width].ljust(sequence_width)} | "
                    f"{str(difference).ljust(token_diff_width)} | "
                    f"{clean_tok1} | "
                    f"{clean_tok2} \n"
                )

    return summary


# Function to check if a token is printable in English (basic ASCII)
def is_printable_ascii(s):
    return all(c in string.printable for c in s)


# Function to check if a token contains Russian, Chinese, or other non-English characters
def is_non_english(s):
    # Regex to match non-Latin (non-English) characters, including Cyrillic and Chinese
    return bool(re.search(r"[^\x00-\x7F]", s))


# Function to check if a token contains math or nonspecific language symbols (like Greek, etc.)
def is_math_symbol(s):
    # Regex to match common math and technical symbols (you can expand this list)
    math_symbols = r"[\u2200-\u22FF\u27C0-\u27EF]"
    return bool(re.search(math_symbols, s))


# Function to write tokens to a single file in a human-readable format
def write_tokens_to_file(vocab, output_path):
    # Sort tokens by their IDs
    sorted_vocab = sorted(vocab.items(), key=lambda item: item[1])

    with open(output_path, "w", encoding="utf-8") as f:
        for token, token_id in sorted_vocab:
            # Decode the token (byte string) to a readable string
            decoded_token = token.decode("utf-8", errors="replace")

            # Determine the label based on the content of the token
            label = ""
            if is_non_english(decoded_token):
                # Non-English characters (e.g., Russian, Chinese) -> Label them explicitly
                label = "[Non-English]"
            elif is_math_symbol(decoded_token):
                # Math and technical symbols -> Label them
                label = "[Math/Technical]"
            elif not is_printable_ascii(decoded_token):
                # Non-printable or unrecognized -> Label them as non-printable
                decoded_token = "<non-printable>"
                label = "[Non-printable]"

            # Format and write each line with a max token length of 32 characters, including labels
            f.write(f"Token: {decoded_token:<32} ID: {token_id} {label}\n")

    print(f"Tokens and IDs have been written to {output_path} (without ANSI color codes)")


def detailed_tokenization_comparison(text: str, tokenizer1, tokenizer2, label1: str, label2: str) -> str:
    """Provides a detailed comparison of tokenization results for specific sequences of interest."""

    tokens1 = encode_text(tokenizer1, text)
    tokens2 = encode_text(tokenizer2, text)

    # decoded1 = [tokenizer1.decode([t]) for t in tokens1]
    # decoded2 = [tokenizer2.decode([t]) for t in tokens2]

    # Identify sequences that might show significant differences
    # interesting_sequences = identify_interesting_sequences(text)
    interesting_sequences = [t[:40] for t in text.split("\n")]

    # Analyze how each tokenizer handles these sequences
    sequence_comparisons = []
    for sequence in interesting_sequences:
        tok1 = tokenizer1.encode(sequence, add_special_tokens=False)
        tok2 = tokenizer2.encode(sequence, add_special_tokens=False)
        sequence_comparisons.append((sequence, tok1, tok2))

    # Sort comparisons by the difference in the number of tokens
    sequence_comparisons.sort(key=lambda x: abs(len(x[1]) - len(x[2])), reverse=True)

    # Generate detailed summary with decoded tokens
    summary = generate_colored_summary(label1, label2, tokens1, tokens2, tokenizer1, tokenizer2, sequence_comparisons)

    return summary


def validate_tokenizers(setting):
    """Validates the tokenizers by running a comparison on a test text."""
    from transformers import PreTrainedTokenizerFast
    from bpeasy.tokenizer import BPEasyTokenizer

    for setting_option in ["cyme", "ascra", "aeolian"]:  # , "
        print(f"Setting: {setting_option}")
        bp_tokenizer = BPEasyTokenizer.from_file(os.path.join(FINAL_LOCATION, f"bp_basis_{setting_option}.json"))
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=os.path.join(FINAL_LOCATION, f"hf_tokenizer_{setting_option}.json")
        )
        assert hf_tokenizer.decode(hf_tokenizer.encode("sequence")) == "sequence"
        assert hf_tokenizer.encode("sequence") == bp_tokenizer.encode("sequence")

        # Test 1: "Huginn" tokenization
        test_huginn = "Huginn _Huginn Huginn     <|begin_header|>Huginn<|end_header|> blabla"
        bp_huginn = [
            bp_tokenizer.decode([i])
            for i in bp_tokenizer.encode(test_huginn, allowed_special={"<|begin_header|>", "<|end_header|>"})
        ]
        hf_huginn = [hf_tokenizer.decode([i]) for i in hf_tokenizer.encode(test_huginn)]

        if bp_huginn == hf_huginn:
            print(hf_huginn)
        else:
            print("Tokenization inconsistent in HF")
            print(test_huginn)
            print(bp_huginn)
            print(hf_huginn)

        # Test 2: "1" vs " 1" distinction
        test_numbers = "1+1=3 and 1 + 2 = 3 and 1+ 3= 5 asdas blabla"
        bp_num = [bp_tokenizer.decode([i]) for i in bp_tokenizer.encode(test_numbers)]
        hf_num = [hf_tokenizer.decode([i]) for i in hf_tokenizer.encode(test_numbers)]

        if bp_num == hf_num:
            print(hf_num)
        else:
            print("Tokenization inconsistent in HF")
            print(test_numbers)
            print(bp_num)
            print(hf_num)

        # Bracket language
        bracket = "0}{\frac{f(x+h) \\{{{{{asdasdasd   asdasdsd232dasd}} asdasd}\\????///}}"
        bp_num = [bp_tokenizer.decode([i]) for i in bp_tokenizer.encode(bracket)]
        hf_num = [hf_tokenizer.decode([i]) for i in hf_tokenizer.encode(bracket)]

        if bp_num == hf_num:
            print(hf_num)
        else:
            print("Tokenization inconsistent in HF")
            print(bracket)
            print(bp_num)
            print(hf_num)

    # llama2_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    # llama3_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    # llama3_tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn_tokenizer_65k")
    tokenizer2 = PreTrainedTokenizerFast(tokenizer_file=os.path.join(FINAL_LOCATION, "hf_tokenizer_cyme.json"))

    bp_tokenizer = BPEasyTokenizer.from_file(os.path.join(FINAL_LOCATION, f"bp_basis_{setting}.json"))
    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(FINAL_LOCATION, f"hf_tokenizer_{setting}.json"))
    with open("scripts/tokenizer_test.txt", "r", encoding="utf-8") as file:
        test_text = file.read()

    comparison_result = detailed_tokenization_comparison(test_text, hf_tokenizer, tokenizer2, setting, "cyme")
    print(comparison_result)
    # print a human-readable vocab to file:
    write_tokens_to_file(bp_tokenizer.vocab, os.path.join(FINAL_LOCATION, "tokenizer_printout.txt"))

    # 2nd try for many-column comparison
    tokenizers = [
        (PreTrainedTokenizerFast(tokenizer_file=os.path.join(FINAL_LOCATION, "hf_tokenizer_cyme.json")), "cyme"),
        (PreTrainedTokenizerFast(tokenizer_file=os.path.join(FINAL_LOCATION, "hf_tokenizer_aeolian.json")), "aeolian"),
        (AutoTokenizer.from_pretrained("tomg-group-umd/huginn_tokenizer_65k"), "hesiod"),
    ]
    summary = detailed_tokenization_comparison2(test_text, tokenizers)

    print(summary)


def upload_tokenizer(setting):
    """Validates the tokenizers by running a comparison on a test text."""
    from transformers import PreTrainedTokenizerFast

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(FINAL_LOCATION, f"hf_tokenizer_{setting}.json"))

    # define extra stuff and upload
    from tokenizers.processors import TemplateProcessing

    # use HF Tokenizers instead
    slow_tokenizer = hf_tokenizer._tokenizer
    slow_tokenizer.post_processor = TemplateProcessing(  # type: ignore
        single="<|begin_text|> $A <|end_text|>",
        pair=None,
        special_tokens=[
            ("<|begin_text|>", slow_tokenizer.token_to_id("<|begin_text|>")),
            ("<|end_text|>", slow_tokenizer.token_to_id("<|end_text|>")),
        ],
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=slow_tokenizer,
        bos_token="<|begin_text|>",
        eos_token="<|end_text|>",
        pad_token="<|pad|>",
    )

    chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|begin_header|>' + message['role'] + '<|end_header|>\n\n'+ message['content'] | trim + '<|end_turn|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|begin_header|>Huginn<|end_header|>\n\n' }}{% else %}{{ '<|end_text|>' }}{% endif %}"
    hf_tokenizer.chat_template = chat_template
    hf_tokenizer.push_to_hub(
        f"tokenizer_{setting}_{hf_tokenizer.vocab_size}", organization="tomg-group-umd", private=True
    )


if __name__ == "__main__":
    setting = "aeolian"
    validate_tokenizers(setting)
    # upload_tokenizer(setting)
