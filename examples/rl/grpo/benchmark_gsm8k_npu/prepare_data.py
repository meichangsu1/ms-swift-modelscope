"""Convert offline GSM8K question/answer Parquets into aligned Swift and verl inputs."""

import argparse
import hashlib
import random
import re
from pathlib import Path

import json
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

INSTRUCTION = r"Solve step by step and put the final answer within \boxed{}."
SYSTEM = "You are a helpful assistant."


def prepare_split(source, split, tokenizer, output_dir, max_prompt_length, seed):
    raw = pq.read_table(source, columns=["question", "answer"]).to_pylist()
    indices = list(range(len(raw)))
    if split == "train":
        random.Random(seed).shuffle(indices)
    swift_rows, verl_rows = [], []
    for index in indices:
        row = raw[index]
        match = re.search(r"####\s*(-?[0-9][0-9.,]*)", row["answer"])
        if match is None:
            raise ValueError(f"{split} row {index} has no #### numeric ground truth.")
        gold = match.group(1).replace(",", "")
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["question"].strip() + "\n" + INSTRUCTION},
        ]
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if len(ids) > max_prompt_length:
            continue
        swift_rows.append({"messages": messages, "solution": gold})
        verl_rows.append(
            {
                "prompt": messages,
                "data_source": "gsm8k_shared",
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {"split": split, "index": index},
            }
        )
    if not swift_rows:
        raise ValueError(f"No {split} rows fit max_prompt_length={max_prompt_length}.")
    json_path = output_dir / f"swift_{split}.jsonl"
    with json_path.open("w", encoding="utf-8") as stream:
        for row in swift_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    parquet_path = output_dir / f"verl_{split}.parquet"
    pq.write_table(pa.Table.from_pylist(verl_rows), parquet_path)
    return {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "total": len(raw),
        "kept": len(swift_rows),
        "dropped_overlong": len(raw) - len(swift_rows),
        "swift_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
        "verl_sha256": hashlib.sha256(parquet_path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_prompt_length != 512:
        parser.error("The launchers use a 512-token prompt limit; keep --max-prompt-length=512.")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": str(args.model.resolve()),
        "max_prompt_length": args.max_prompt_length,
        "seed": args.seed,
        "instruction": INSTRUCTION,
        "enable_thinking": False,
    }
    for split, source in (("train", args.train_file), ("test", args.test_file)):
        manifest[split] = prepare_split(source, split, tokenizer, args.output_dir, args.max_prompt_length, args.seed)
        print(f"{split}: {manifest[split]['kept']}/{manifest[split]['total']} rows kept")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
