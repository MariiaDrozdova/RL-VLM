import logging
from datasets import Dataset, DatasetDict, load_dataset
import datasets
import random
from torch.utils.data import Dataset as TorchDataset
import re
from collections import Counter

from .config import TRAIN_DATASET_PATH, VAL_DATASET_PATH, TEST_DATASET_PATH, TEST_HARD_NUMBER_DATASET_PATH, TRAIN_SUBSET_SIZE, TEST_SUBSET_SIZE, TEST_HARD_SUBSET_SIZE

logger = logging.getLogger(__name__)

def load_and_merge_datasets():
    """
    Loads and merges separately saved train, validation, and test datasets.
    """
    logger.info(f"Loading datasets from disk... {TRAIN_DATASET_PATH}")
    train_dataset = Dataset.load_from_disk(TRAIN_DATASET_PATH)
    val_dataset = Dataset.load_from_disk(VAL_DATASET_PATH)
    test_dataset = Dataset.load_from_disk(TEST_DATASET_PATH)
    test_dataset_hard_number = Dataset.load_from_disk(TEST_HARD_NUMBER_DATASET_PATH)
    
    # Optionally, select a subset from training data
    if TRAIN_SUBSET_SIZE:
        train_dataset = train_dataset.select(range(TRAIN_SUBSET_SIZE))
    if TEST_SUBSET_SIZE:
        test_dataset = test_dataset.select(range(TEST_SUBSET_SIZE))
    if TEST_HARD_SUBSET_SIZE:
        test_dataset_hard_number = test_dataset_hard_number.select(range(TEST_HARD_SUBSET_SIZE))
    
    logger.info("Datasets loaded and merged.")
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
        "test_hard_number": test_dataset_hard_number
    })

def load_json_datasets():
    """
    Loads JSON datasets if needed.
    """
    logger.info("Loading JSON datasets...")
    data = load_dataset(
        "json",
        data_files={
            "train": "train.json",
            "validation": "validation.json",
            "test": "test.json",
        }
    )
    logger.info("JSON datasets loaded.")
    return data

def extract_vlm_data(text):
    """
    From a formatted VLM response, pull out:
      - the raw reasoning between <think>…</think>
      - the final answer between <answer>…</answer>
    Returns (reasoning:str, answer:str) or (None,None) if tags missing.
    """
    think_m = re.search(r"<think>(.*?)</think>", text, re.S)
    ans_m   = re.search(r"<answer>(.*?)</answer>", text, re.S)
    reasoning = think_m.group(1).strip() if think_m else None
    answer    = ans_m.group(1).strip()   if ans_m   else None
    return reasoning, answer

def format_vlm_response(full_description_s: str, answer_s: str, answers_options: str) -> str:
    """
    Given a description string of circles and an answer string, returns a styled response
    with <think></think> tokens containing step-by-step reasoning, and <answer></answer>
    wrapping the final answer. Useful for training vision-language models (VLMs).

    Args:
        full_description_s: A single string containing descriptions of circles, e.g.
            "The circle of color cyan and size 8 is at coordinates (118,100), The circle of color white and size 9 is at coordinates (120,54), ..."
        answer_s: The final answer string, e.g. "even" or "odd".

    Returns:
        A formatted string with chain-of-thought reasoning and final answer tags.
    """
    # Regex to extract properties
    pattern = r"The circle of color (?P<color>\w+) and size (?P<size>\d+) is at coordinates \((?P<x>\d+),(?P<y>\d+)\)"
    matches = list(re.finditer(pattern, full_description_s))
    
    # Parse into list of dicts
    circles = [m.groupdict() for m in matches]
    
    # Build chain-of-thought lines
    lines = []
    lines.append("We have the following circles:")
    for c in circles:
        lines.append(f"- Color {c['color']}, size {c['size']}, at ({c['x']},{c['y']})")
    
    # Count by color
    color_counts = Counter(c['color'] for c in circles)

    lines.append("\nCounts by color:")
    for color, count in color_counts.items():
        lines.append(f"- {color}: {count}")
    
    # Example reasoning: check parity of white circles
    if answers_options == "shortened":
        lines = []
    total_count = sum(list(color_counts.values()))
    lines.append(f"\nThere are {total_count} circles.")
    if total_count % 2 == 0:
        lines.append("Since the number of circles is even, the answer is 'even'.")
    else:
        lines.append("Since the number of circles is odd, the answer is 'odd'.")
    
    # Wrap the reasoning and answer
    think_block = "<think>\n" + "\n".join(lines) + "\n</think>"
    answer_block = f"<answer>{answer_s}</answer>"
    return think_block + "\n" + answer_block

class CirclesQADataset(TorchDataset):
    def __init__(self, hf_dataset, prefix, answers_options="full"):
        super().__init__()              # ensure base class inits
        self.hf_dataset = hf_dataset    # keep HFDataset here
        self.prefix = prefix
        self.answers_options = answers_options

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        question_id = 8  # or whichever you prefer
        answer_s = example['answers'][8]
        full_description_s = example['answers'][9]
        answer = format_vlm_response(full_description_s, answer_s, self.answers_options)
        if self.prefix == "CAPTION":
            question = self.prefix 
        else:
            question = self.prefix + example['questions'][question_id]

        image = example['image']
        if image.mode != "RGB":
            image = image.convert("RGB")
        return question, answer, image