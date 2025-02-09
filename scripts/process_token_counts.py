import re
from collections import defaultdict
from pydantic import BaseModel
from typing import List, Optional
import yaml


class DataSource(BaseModel):
    address: str
    subset: Optional[str] = None
    features: List[str]
    needs_chat_templating: bool = False
    license: Optional[str] = None
    citation: Optional[str] = None
    machine_generated: bool = False
    requires_software_heritage_aws_download: Optional[bool] = False
    weight: float = 1.0
    category: str = "generic-web"
    every_token_is_sacred: bool = False

    def fill_empty_citation(self):
        if not self.citation:
            self.citation = f"https://huggingface.co/datasets/{self.address}"

    def fill_empty_license(self):
        if not self.license:
            self.license = "other"


class DataSources(BaseModel):
    sources: dict[str, DataSource]

    @classmethod
    def from_yaml(cls, file_path: str) -> "DataSources":
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        sources = {name: DataSource(**source_data) for name, source_data in data.items()}
        for source in sources.values():
            source.fill_empty_citation()
            source.fill_empty_license()
        return cls(sources=sources)


def extract_tokens(log_line):
    # Updated the regex to match the format --> <chosen_tokens> -- <total_tokens> -- <character_count>
    match = re.search(r"--->\s*(\d+)\s*--\s*(\d+)\s*--\s*(\d+)", log_line)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return 0, 0, 0


def count_tokens():
    tokens_per_dataset = {}
    total_tokens_per_dataset = {}
    tokens_per_category = defaultdict(lambda: 0)
    datasets_per_category = defaultdict(list)
    total_chosen_tokens_counted = 0

    data_sources = DataSources.from_yaml("scripts/sources.yaml")
    with open("scripts/combined_logs.txt", "r") as f:
        logs = f.read().split("==> tokenizecount_")

    for idx, (name, source) in enumerate(data_sources.sources.items()):
        # Use the extract_tokens function to get the token count from the correct log
        try:
            token_count, total_tokens, _ = extract_tokens(logs[idx + 1])  # Adjust for the correct log index
        except Exception:
            token_count, total_tokens, _ = 0, 0, 0

        tokens_per_dataset[name] = token_count
        total_tokens_per_dataset[name] = total_tokens
        tokens_per_category[source.category] += token_count
        datasets_per_category[source.category].append((name, token_count))
        total_chosen_tokens_counted += token_count

    # Printout with additional information
    print("\nToken counts per file:")
    for idx, (name, count) in enumerate(tokens_per_dataset.items()):
        total_tokens = total_tokens_per_dataset[name]
        incinerated_percentage = ((total_tokens - count) / total_tokens) * 100 if total_tokens != 0 else float("NaN")
        print(f"{idx:>3}: {name:<28}: {count/1e9:6.2f}B (Incinerated: {incinerated_percentage:.2f}%)")

    print("\nToken counts per category:")
    for category, count in sorted(tokens_per_category.items(), key=lambda x: x[1], reverse=True):
        print(f"{category}: {count/1e9:.2f}B ({count/total_chosen_tokens_counted*100:.2f}%)")
        print("Constituent datasets:")
        for dataset, tokens in sorted(datasets_per_category[category], key=lambda x: x[1], reverse=True):
            if tokens > 0:
                print(f"  - {dataset}: {tokens/1e9:.2f}B ({tokens/count*100:.2f}%)")
        print()

    print(f"\nTotal tokens: {total_chosen_tokens_counted/1e12:.2f}T")


if __name__ == "__main__":
    count_tokens()
