#!/usr/bin/env python3
"""Convert TextVQA train Parquet shards to MiniMind-V SFT format."""

import argparse
import io
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Image as DatasetImage
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm


NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
ARTICLES = {"a", "an", "the"}
PROMPT_TEMPLATE = (
    "<image>\n{question}\n"
    "Answer the question using a single word or phrase."
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("image_bytes", pa.binary(), nullable=False),
        pa.field("conversations", pa.string(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        pa.field("image_id", pa.string(), nullable=False),
    ]
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert TextVQA train shards to MiniMind-V SFT Parquet."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/data/minimind-v/textvqa/data"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/data/minimind-v/textvqa/processed/"
            "textvqa_train_sft.parquet"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N rows for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser.parse_args()


def validate_args(args):
    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"TextVQA directory not found: {args.input_dir}"
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. "
            "Use --overwrite to replace it."
        )


def find_train_shards(input_dir):
    patterns = ("train-*.parquet", "train_*.parquet")
    files = {
        path.resolve()
        for pattern in patterns
        for path in input_dir.rglob(pattern)
        if path.is_file()
    }
    if not files:
        available = sorted(
            str(path.relative_to(input_dir))
            for path in input_dir.rglob("*.parquet")
        )
        raise FileNotFoundError(
            "No train Parquet shards were found. "
            f"Available Parquet files: {available[:30]}"
        )
    return sorted(files)


def normalize_answer(answer):
    text = str(answer).lower().strip()
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    text = re.sub(r"[^\w\s:$.'%-]", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    words = []
    for word in text.split():
        word = NUMBER_WORDS.get(word, word)
        if word not in ARTICLES:
            words.append(word)
    return " ".join(words)


def choose_target_answer(answers):
    if answers is None:
        answers = []
    if hasattr(answers, "tolist"):
        answers = answers.tolist()
    normalized = [
        normalize_answer(answer)
        for answer in list(answers)
    ]
    normalized = [answer for answer in normalized if answer]
    if not normalized:
        return None

    counts = Counter(normalized)
    highest_count = max(counts.values())
    for answer in normalized:
        if counts[answer] == highest_count:
            return answer
    return None


def encode_rgb_image(image, image_format="PNG"):
    buffer = io.BytesIO()
    rgb_image = image.convert("RGB")
    if image_format in {"JPEG", "JPG"}:
        rgb_image.save(
            buffer,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
    else:
        rgb_image.save(buffer, format="PNG")
    return buffer.getvalue()


def ensure_rgb_bytes(content):
    content = bytes(content)
    with Image.open(io.BytesIO(content)) as image:
        if image.mode == "RGB":
            return content
        return encode_rgb_image(image, image.format or "PNG")


def image_to_bytes(image_value, input_dir):
    if isinstance(image_value, (bytes, bytearray, memoryview)):
        return ensure_rgb_bytes(image_value)

    if isinstance(image_value, dict):
        content = image_value.get("bytes")
        if content is not None:
            return ensure_rgb_bytes(content)
        path_value = image_value.get("path")
        if path_value:
            path = Path(path_value)
            if not path.is_absolute():
                path = input_dir / path
            return ensure_rgb_bytes(path.read_bytes())

    if isinstance(image_value, Image.Image):
        return encode_rgb_image(image_value)

    if isinstance(image_value, (str, os.PathLike)):
        path = Path(image_value)
        if not path.is_absolute():
            path = input_dir / path
        return ensure_rgb_bytes(path.read_bytes())

    raise TypeError(
        f"Unsupported image field type: {type(image_value).__name__}"
    )


def build_conversation(question, answer):
    messages = [
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                question=str(question).strip()
            ),
        },
        {"role": "assistant", "content": answer},
    ]
    return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))


def build_output_row(sample, input_dir):
    answer = choose_target_answer(sample["answers"])
    if answer is None:
        return None
    return {
        "image_bytes": image_to_bytes(sample["image"], input_dir),
        "conversations": build_conversation(
            sample["question"],
            answer,
        ),
        "question_id": str(sample["question_id"]),
        "image_id": str(sample.get("image_id", "")),
    }


def write_batch(writer, rows):
    columns = {
        name: [row[name] for row in rows]
        for name in OUTPUT_SCHEMA.names
    }
    writer.write_table(pa.Table.from_pydict(columns, schema=OUTPUT_SCHEMA))


def convert_dataset(dataset, input_dir, output, batch_size):
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    rows = []
    written_rows = 0
    skipped_question_ids = []

    with pq.ParquetWriter(
        temporary_output,
        OUTPUT_SCHEMA,
        compression="zstd",
    ) as writer:
        for sample in tqdm(dataset, desc="Converting TextVQA", unit="row"):
            row = build_output_row(sample, input_dir)
            if row is None:
                skipped_question_ids.append(str(sample["question_id"]))
                continue
            rows.append(row)
            written_rows += 1
            if len(rows) == batch_size:
                write_batch(writer, rows)
                rows.clear()
        if rows:
            write_batch(writer, rows)

    if written_rows == 0:
        raise RuntimeError("No supervised training rows were produced.")
    os.replace(temporary_output, output)
    return written_rows, skipped_question_ids


def verify_output(output, expected_rows):
    parquet_file = pq.ParquetFile(output)
    if parquet_file.metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"Row count mismatch: expected {expected_rows}, "
            f"got {parquet_file.metadata.num_rows}."
        )
    if parquet_file.schema_arrow != OUTPUT_SCHEMA:
        raise RuntimeError(
            "Output schema mismatch: "
            f"{parquet_file.schema_arrow}"
        )

    sample = (
        parquet_file.read_row_group(0)
        .slice(0, 1)
        .to_pylist()[0]
    )
    Image.open(io.BytesIO(sample["image_bytes"])).verify()
    conversations = json.loads(sample["conversations"])
    if [message["role"] for message in conversations] != [
        "user",
        "assistant",
    ]:
        raise RuntimeError("Invalid conversation roles in the output.")
    if "<image>" not in conversations[0]["content"]:
        raise RuntimeError("The user prompt does not contain <image>.")
    if not conversations[1]["content"]:
        raise RuntimeError("The assistant answer is empty.")


def main():
    args = parse_args()
    validate_args(args)
    train_shards = find_train_shards(args.input_dir)

    print("TextVQA train shards:")
    for path in train_shards:
        print(f"  {path}")

    dataset = load_dataset(
        "parquet",
        data_files={"train": [str(path) for path in train_shards]},
        split="train",
    )
    required_columns = {
        "image",
        "question",
        "answers",
        "question_id",
    }
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(
            f"Missing TextVQA columns: {sorted(missing_columns)}. "
            f"Available columns: {dataset.column_names}"
        )

    dataset = dataset.cast_column("image", DatasetImage(decode=False))
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written_rows, skipped_question_ids = convert_dataset(
        dataset,
        args.input_dir,
        args.output,
        args.batch_size,
    )
    verify_output(args.output, written_rows)

    print(f"Saved: {args.output}")
    print(f"Input rows: {len(dataset)}")
    print(f"Written rows: {written_rows}")
    print(f"Skipped rows without answers: {len(skipped_question_ids)}")
    if skipped_question_ids:
        print(
            "First skipped question IDs: "
            + ", ".join(skipped_question_ids[:20])
        )
    print(f"Size: {args.output.stat().st_size / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
