#!/usr/bin/env python3
"""
Evaluate a native PyTorch MiniMind-V checkpoint on local TextVQA Parquet shards.

Supported modes:

* validation/train: write per-sample predictions and compute TextVQA/VQA soft
  accuracy locally.
* test: write an EvalAI-compatible submission JSON. TextVQA test labels are not
  public, so no local accuracy is reported.

The script is intended to live at:

    <minimind-v-repo>/eval/textvqa_eval.py

It deliberately does not use OCR tokens from TextVQA. The model receives only
the image and question, which keeps comparisons between P32/P16/resampler
variants controlled.
"""

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from datasets import Image as HFImage
from datasets import load_dataset
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.model_vlm import MiniMindVLM, VLMConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MiniMind-V on local TextVQA Parquet shards. "
            "Use validation for local accuracy and test for EvalAI submission."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/minimind-v/textvqa/data"),
        help="Directory containing TextVQA Parquet shards.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "val", "test"),
        default="validation",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/minimind-v/out/sft_vlm_768.pth"),
        help="Native MiniMind-V PyTorch checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "model",
        help="Directory containing the MiniMind tokenizer files.",
    )
    parser.add_argument(
        "--vision-model",
        type=Path,
        default=Path("/data/minimind-v/model/siglip2-base-p32-256-ve"),
        help="Local SigLIP2 model and image-processor directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Run directory. Defaults to a timestamped directory.",
    )
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--image-token-len", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=(
            "<image>\n{question}\n"
            "Answer the question using a single word or phrase."
        ),
        help="Must contain exactly one <image> and one {question}.",
    )
    parser.add_argument(
        "--preserve-question-case",
        action="store_true",
        help=(
            "Do not apply Python str.capitalize() to the question. "
            "The default matches lmms-eval's TextVQA prompt."
        ),
    )
    parser.add_argument("--open-thinking", action="store_true")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start from this dataset index before applying --limit.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only this many examples. Use only for smoke tests.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from predictions.jsonl in an existing output directory.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record sample errors and continue. Default behavior is fail-fast.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="fsync predictions.jsonl after this many new records.",
    )
    parser.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help=(
            "Fail if load_state_dict reports non-vision missing keys or any "
            "unexpected keys."
        ),
    )
    parser.add_argument(
        "--skip-checkpoint-hash",
        action="store_true",
        help="Skip checkpoint SHA-256 calculation.",
    )
    args = parser.parse_args()

    if args.split == "val":
        args.split = "validation"

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = (
            Path("/data/minimind-v/out/eval_runs")
            / f"{args.checkpoint.stem}-textvqa-{args.split}-{timestamp}"
        )
    return args


def validate_args(args):
    if not args.dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory not found: {args.dataset_dir}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.tokenizer_path.is_dir():
        raise NotADirectoryError(
            f"Tokenizer directory not found: {args.tokenizer_path}"
        )
    if not args.vision_model.is_dir():
        raise NotADirectoryError(
            f"Vision-model directory not found: {args.vision_model}"
        )
    if not args.device.startswith("cuda"):
        raise ValueError("This evaluator currently requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current PyTorch environment.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16.")
    if args.image_token_len < 1:
        raise ValueError("--image-token-len must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")
    if args.flush_every < 1:
        raise ValueError("--flush-every must be at least 1.")
    if args.prompt_template.count("<image>") != 1:
        raise ValueError("--prompt-template must contain exactly one <image>.")
    if args.prompt_template.count("{question}") != 1:
        raise ValueError("--prompt-template must contain exactly one {question}.")

    predictions_path = args.output_dir / "predictions.jsonl"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError(
                "Output directory is non-empty. Choose a new directory or use "
                f"--resume: {args.output_dir}"
            )
        if not predictions_path.is_file():
            raise FileNotFoundError(
                "--resume requires an existing predictions.jsonl: "
                f"{predictions_path}"
            )


def resolve_parquet_files(dataset_dir, split):
    aliases = {
        "train": ("train",),
        "validation": ("validation", "val"),
        "test": ("test",),
    }[split]
    matches = set()
    for alias in aliases:
        matches.update(dataset_dir.rglob(f"{alias}-*.parquet"))
        matches.update(dataset_dir.rglob(f"{alias}_*.parquet"))
    files = sorted(path.resolve() for path in matches if path.is_file())
    if not files:
        available = sorted(path.name for path in dataset_dir.rglob("*.parquet"))
        raise FileNotFoundError(
            f"No Parquet shards found for split={split} under {dataset_dir}. "
            f"Available files: {available[:30]}"
        )
    return files


def load_textvqa_dataset(files, split):
    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in files]},
        split=split,
    )
    required = {"image", "question", "question_id"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"TextVQA Parquet is missing required columns: {sorted(missing)}. "
            f"Columns: {dataset.column_names}"
        )
    if split != "test" and "answers" not in dataset.column_names:
        raise ValueError(
            f"Split {split} requires an answers column for local scoring."
        )
    try:
        dataset = dataset.cast_column("image", HFImage(decode=True))
    except (TypeError, ValueError):
        # Some converted Parquet datasets already expose PIL images correctly,
        # while others need manual bytes/path decoding in decode_image().
        pass
    return dataset


def set_deterministic_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_dtype(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def normalize_state_dict_keys(state_dict):
    normalized = {}
    for name, value in state_dict.items():
        if "mask" in name:
            continue
        while name.startswith("module.") or name.startswith("_orig_mod."):
            if name.startswith("module."):
                name = name[len("module.") :]
            elif name.startswith("_orig_mod."):
                name = name[len("_orig_mod.") :]
        normalized[name] = value
    return normalized


def load_model(args, device, dtype):
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer_path))
    config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        image_token_len=args.image_token_len,
    )
    model = MiniMindVLM(config, vision_model_path=str(args.vision_model))
    if model.vision_encoder is None or model.processor is None:
        raise RuntimeError(
            f"Failed to load SigLIP2 model or processor from {args.vision_model}"
        )

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    if not isinstance(state_dict, dict):
        raise TypeError(
            "Checkpoint must be a state_dict or contain a dictionary under 'model'."
        )
    state_dict = normalize_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)

    missing_keys = list(incompatible.missing_keys)
    unexpected_keys = list(incompatible.unexpected_keys)
    nonvision_missing = [
        name
        for name in missing_keys
        if not name.startswith("vision_encoder.")
    ]
    if args.strict_checkpoint and (nonvision_missing or unexpected_keys):
        raise RuntimeError(
            "Checkpoint is incompatible with the requested model config. "
            f"Non-vision missing keys: {nonvision_missing}; "
            f"unexpected keys: {unexpected_keys}"
        )
    if nonvision_missing or unexpected_keys:
        print(
            "WARNING: checkpoint loaded with incompatible keys. "
            "Review resolved_config.json before reporting results.",
            file=sys.stderr,
        )
        print(
            json.dumps(
                {
                    "nonvision_missing_keys": nonvision_missing,
                    "unexpected_keys": unexpected_keys,
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )

    model = model.eval().to(device=device, dtype=dtype)
    return model, tokenizer, model.processor, {
        "missing_keys": missing_keys,
        "nonvision_missing_keys": nonvision_missing,
        "unexpected_keys": unexpected_keys,
    }


def move_processor_output(processor_output, device, dtype):
    return {
        name: tensor.to(
            device=device,
            dtype=dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in processor_output.items()
    }


def decode_image(value, dataset_dir):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image:
            return image.convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")
        image_path = value.get("path")
        if image_path:
            path = Path(image_path)
            if not path.is_absolute():
                path = dataset_dir / path
            with Image.open(path) as image:
                return image.convert("RGB")
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if not path.is_absolute():
            path = dataset_dir / path
        with Image.open(path) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value)!r}")


def apply_chat_template(tokenizer, messages, open_thinking):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            open_thinking=open_thinking,
        )
    except TypeError:
        if open_thinking:
            raise RuntimeError(
                "The tokenizer chat template does not support open_thinking=True."
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def prepare_sample(sample, model, tokenizer, processor, args, device, dtype):
    image = decode_image(sample["image"], args.dataset_dir)
    processor_output = MiniMindVLM.image2tensor(image, processor)
    pixel_values = move_processor_output(processor_output, device, dtype)

    question = str(sample["question"]).strip()
    prompted_question = (
        question
        if args.preserve_question_case
        else question.capitalize()
    )
    prompt = args.prompt_template.format(question=prompted_question)
    prompt = prompt.replace(
        "<image>",
        model.config.image_special_token * model.config.image_token_len,
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = apply_chat_template(
        tokenizer,
        messages,
        open_thinking=args.open_thinking,
    )
    tokenized = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
    )
    inputs = {
        name: tensor.to(device)
        for name, tensor in tokenized.items()
    }
    return inputs, pixel_values, question


@torch.inference_mode()
def generate_answer(model, tokenizer, inputs, pixel_values, args):
    generation_kwargs = {
        "inputs": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": pixel_values,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
    }
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    generated_ids = model.generate(**generation_kwargs)
    prompt_length = inputs["input_ids"].shape[-1]
    answer_ids = generated_ids[0, prompt_length:]
    answer = tokenizer.decode(
        answer_ids,
        skip_special_tokens=True,
    ).strip()
    return answer, int(prompt_length), int(answer_ids.numel())


class EvalAIAnswerProcessor:
    """TextVQA/VQA answer normalization compatible with EvalAI/lmms-eval."""

    CONTRACTIONS = {
        "aint": "ain't",
        "arent": "aren't",
        "cant": "can't",
        "couldve": "could've",
        "couldnt": "couldn't",
        "couldn'tve": "couldn't've",
        "couldnt've": "couldn't've",
        "didnt": "didn't",
        "doesnt": "doesn't",
        "dont": "don't",
        "hadnt": "hadn't",
        "hadnt've": "hadn't've",
        "hadn'tve": "hadn't've",
        "hasnt": "hasn't",
        "havent": "haven't",
        "hed": "he'd",
        "hed've": "he'd've",
        "he'dve": "he'd've",
        "hes": "he's",
        "howd": "how'd",
        "howll": "how'll",
        "hows": "how's",
        "id've": "i'd've",
        "i'dve": "i'd've",
        "im": "i'm",
        "ive": "i've",
        "isnt": "isn't",
        "itd": "it'd",
        "itd've": "it'd've",
        "it'dve": "it'd've",
        "itll": "it'll",
        "let's": "let's",
        "maam": "ma'am",
        "mightnt": "mightn't",
        "mightnt've": "mightn't've",
        "mightn'tve": "mightn't've",
        "mightve": "might've",
        "mustnt": "mustn't",
        "mustve": "must've",
        "neednt": "needn't",
        "notve": "not've",
        "oclock": "o'clock",
        "oughtnt": "oughtn't",
        "shant": "shan't",
        "shed've": "she'd've",
        "she'dve": "she'd've",
        "shes": "she's",
        "shouldve": "should've",
        "shouldnt": "shouldn't",
        "shouldnt've": "shouldn't've",
        "shouldn'tve": "shouldn't've",
        "somebody'd": "somebodyd",
        "somebodyd've": "somebody'd've",
        "somebody'dve": "somebody'd've",
        "somebodyll": "somebody'll",
        "somebodys": "somebody's",
        "someoned": "someone'd",
        "someoned've": "someone'd've",
        "someone'dve": "someone'd've",
        "someonell": "someone'll",
        "someones": "someone's",
        "somethingd": "something'd",
        "somethingd've": "something'd've",
        "something'dve": "something'd've",
        "somethingll": "something'll",
        "thats": "that's",
        "thered": "there'd",
        "thered've": "there'd've",
        "there'dve": "there'd've",
        "therere": "there're",
        "theres": "there's",
        "theyd": "they'd",
        "theyd've": "they'd've",
        "they'dve": "they'd've",
        "theyll": "they'll",
        "theyre": "they're",
        "theyve": "they've",
        "twas": "'twas",
        "wasnt": "wasn't",
        "wed've": "we'd've",
        "we'dve": "we'd've",
        "weve": "we've",
        "werent": "weren't",
        "whatll": "what'll",
        "whatre": "what're",
        "whats": "what's",
        "whatve": "what've",
        "whens": "when's",
        "whered": "where'd",
        "wheres": "where's",
        "whereve": "where've",
        "whod": "who'd",
        "whod've": "who'd've",
        "who'dve": "who'd've",
        "wholl": "who'll",
        "whos": "who's",
        "whove": "who've",
        "whyll": "why'll",
        "whyre": "why're",
        "whys": "why's",
        "wont": "won't",
        "wouldve": "would've",
        "wouldnt": "wouldn't",
        "wouldnt've": "wouldn't've",
        "wouldn'tve": "wouldn't've",
        "yall": "y'all",
        "yall'll": "y'all'll",
        "y'allll": "y'all'll",
        "yall'd've": "y'all'd've",
        "y'alld've": "y'all'd've",
        "y'all'dve": "y'all'd've",
        "youd": "you'd",
        "youd've": "you'd've",
        "you'dve": "you'd've",
        "youll": "you'll",
        "youre": "you're",
        "youve": "you've",
    }
    NUMBER_MAP = {
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
    PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
    COMMA_STRIP = re.compile(r"(?<=\d)(,)+(?=\d)")
    PUNCTUATIONS = (
        ";",
        r"/",
        "[",
        "]",
        '"',
        "{",
        "}",
        "(",
        ")",
        "=",
        "+",
        "\\",
        "_",
        "-",
        ">",
        "<",
        "@",
        "`",
        ",",
        "?",
        "!",
    )

    def process_punctuation(self, text):
        output = text
        for punctuation in self.PUNCTUATIONS:
            if (
                punctuation + " " in text
                or " " + punctuation in text
                or self.COMMA_STRIP.search(text) is not None
            ):
                output = output.replace(punctuation, "")
            else:
                output = output.replace(punctuation, " ")
        return self.PERIOD_STRIP.sub("", output)

    def process_digit_article(self, text):
        output = []
        for word in text.lower().split():
            word = self.NUMBER_MAP.get(word, word)
            if word not in self.ARTICLES:
                output.append(self.CONTRACTIONS.get(word, word))
        return " ".join(output)

    def __call__(self, value):
        text = str(value).lower()
        text = text.replace(",", "").replace("?", "").replace("'s", " 's")
        text = text.replace("\n", " ").replace("\t", " ").strip()
        text = self.process_punctuation(text)
        return self.process_digit_article(text)


def textvqa_soft_accuracy(prediction, answers, answer_processor):
    normalized_prediction = answer_processor(prediction)
    normalized_answers = [answer_processor(answer) for answer in answers]
    if not normalized_answers:
        raise ValueError("No reference answers were provided.")

    per_reference_scores = []
    for index in range(len(normalized_answers)):
        other_answers = (
            normalized_answers[:index] + normalized_answers[index + 1 :]
        )
        matches = sum(
            answer == normalized_prediction
            for answer in other_answers
        )
        per_reference_scores.append(min(1.0, matches / 3.0))
    return statistics.mean(per_reference_scores), normalized_prediction


def normalize_answers(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, list)):
        return [str(answer) for answer in value]
    return [str(value)]


def json_safe_scalar(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
        output_file.flush()
        os.fsync(output_file.fileno())
    os.replace(temporary_path, path)


def load_existing_predictions(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {error}"
                ) from error
    return records


def append_prediction(output_file, record, should_sync):
    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_file.flush()
    if should_sync:
        os.fsync(output_file.fileno())


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def run_git_command(arguments):
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def build_environment(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
    }


def build_indices(dataset_length, start_index, limit):
    if start_index >= dataset_length:
        raise ValueError(
            f"--start-index {start_index} is outside dataset length "
            f"{dataset_length}."
        )
    stop = dataset_length
    if limit is not None:
        stop = min(stop, start_index + limit)
    return list(range(start_index, stop))


def is_unlabeled_answers(answers):
    return not answers or all(not str(answer).strip() for answer in answers)


def evaluate(args, dataset, model, tokenizer, processor, device, dtype):
    predictions_path = args.output_dir / "predictions.jsonl"
    existing_records = (
        load_existing_predictions(predictions_path)
        if args.resume
        else []
    )
    completed_indices = {
        int(record["dataset_index"])
        for record in existing_records
        if record.get("status") == "ok"
    }
    records = list(existing_records)
    answer_processor = EvalAIAnswerProcessor()
    indices = build_indices(len(dataset), args.start_index, args.limit)
    pending_indices = [
        index for index in indices if index not in completed_indices
    ]

    new_record_count = 0
    mode = "a" if args.resume else "w"
    with predictions_path.open(mode, encoding="utf-8") as output_file:
        progress = tqdm(
            pending_indices,
            desc=f"TextVQA {args.split}",
            unit="sample",
        )
        for dataset_index in progress:
            sample = dataset[dataset_index]
            question_id = json_safe_scalar(sample["question_id"])
            started_at = time.perf_counter()
            try:
                inputs, pixel_values, question = prepare_sample(
                    sample,
                    model,
                    tokenizer,
                    processor,
                    args,
                    device,
                    dtype,
                )
                raw_prediction, prompt_tokens, generated_tokens = generate_answer(
                    model,
                    tokenizer,
                    inputs,
                    pixel_values,
                    args,
                )
                normalized_prediction = answer_processor(raw_prediction)
                answers = normalize_answers(sample.get("answers"))
                record = {
                    "dataset_index": dataset_index,
                    "question_id": question_id,
                    "image_id": json_safe_scalar(sample.get("image_id")),
                    "question": question,
                    "raw_prediction": raw_prediction,
                    "prediction": normalized_prediction,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "status": "ok",
                }
                if args.split != "test":
                    if is_unlabeled_answers(answers):
                        raise ValueError(
                            f"Sample {question_id} has no public reference answers."
                        )
                    score, normalized_prediction = textvqa_soft_accuracy(
                        raw_prediction,
                        answers,
                        answer_processor,
                    )
                    record["prediction"] = normalized_prediction
                    record["answers"] = answers
                    record["score"] = score

                del inputs, pixel_values
            except Exception as error:
                record = {
                    "dataset_index": dataset_index,
                    "question_id": question_id,
                    "question": str(sample.get("question", "")),
                    "status": "error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "elapsed_seconds": time.perf_counter() - started_at,
                }
                if not args.continue_on_error:
                    append_prediction(output_file, record, should_sync=True)
                    raise

            records.append(record)
            new_record_count += 1
            append_prediction(
                output_file,
                record,
                should_sync=new_record_count % args.flush_every == 0,
            )
            if record["status"] == "ok" and args.split != "test":
                progress.set_postfix(
                    accuracy=f"{record['score'] * 100:.2f}%"
                )

        output_file.flush()
        os.fsync(output_file.fileno())
    return records, indices


def finalize_results(args, records, selected_indices):
    selected_index_set = set(selected_indices)
    selected_records = [
        record
        for record in records
        if int(record["dataset_index"]) in selected_index_set
    ]
    # If a resumed file contains duplicate successful records, retain the last
    # one for deterministic final metrics/submission.
    latest_by_index = {}
    for record in selected_records:
        latest_by_index[int(record["dataset_index"])] = record
    selected_records = [
        latest_by_index[index]
        for index in sorted(latest_by_index)
    ]

    successful = [
        record for record in selected_records if record["status"] == "ok"
    ]
    errors = [
        record for record in selected_records if record["status"] != "ok"
    ]
    complete = (
        len(successful) == len(selected_indices)
        and not errors
    )
    metrics = {
        "split": args.split,
        "selected_example_count": len(selected_indices),
        "successful_prediction_count": len(successful),
        "error_count": len(errors),
        "complete": complete,
        "local_accuracy_available": args.split != "test",
    }

    if args.split == "test":
        submission = [
            {
                "question_id": record["question_id"],
                "answer": record["prediction"],
            }
            for record in successful
        ]
        atomic_write_json(args.output_dir / "submission.json", submission)
        metrics["note"] = (
            "TextVQA test answers are not public. Upload submission.json "
            "to the official EvalAI challenge to obtain the test score."
        )
    else:
        scores = [float(record["score"]) for record in successful]
        if scores:
            accuracy = statistics.mean(scores)
            metrics["accuracy"] = accuracy
            metrics["accuracy_percent"] = accuracy * 100.0
            metrics["scored_example_count"] = len(scores)
        if not complete:
            metrics["note"] = (
                "The run is incomplete; accuracy is over successful samples "
                "only and must not be reported as a full-split result."
            )

    atomic_write_json(args.output_dir / "metrics.json", metrics)
    return metrics


def main():
    args = parse_args()
    validate_args(args)
    parquet_files = resolve_parquet_files(args.dataset_dir, args.split)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = resolve_dtype(args.dtype)
    set_deterministic_seed(args.seed)

    print("Loading TextVQA Parquet shards:")
    for path in parquet_files:
        print(f"  {path}")
    dataset = load_textvqa_dataset(parquet_files, args.split)
    print(f"Loaded split={args.split}, rows={len(dataset)}")

    model, tokenizer, processor, checkpoint_load = load_model(
        args,
        device,
        dtype,
    )
    checkpoint_hash = (
        None
        if args.skip_checkpoint_hash
        else sha256_file(args.checkpoint)
    )
    resolved_config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "repository_root": str(REPO_ROOT),
        "repository_commit": run_git_command(["rev-parse", "HEAD"]),
        "repository_diff_stat": run_git_command(["diff", "--stat"]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_load": checkpoint_load,
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "vision_model_path": str(args.vision_model.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "dataset_split": args.split,
        "dataset_rows": len(dataset),
        "parquet_files": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            for path in parquet_files
        ],
        "model_config": {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": args.use_moe,
            "image_token_len": args.image_token_len,
            "dtype": args.dtype,
        },
        "evaluation_config": {
            "prompt_template": args.prompt_template,
            "preserve_question_case": args.preserve_question_case,
            "open_thinking": args.open_thinking,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "decoding": "greedy",
            "uses_textvqa_ocr_tokens": False,
            "start_index": args.start_index,
            "limit": args.limit,
            "resume": args.resume,
        },
        "environment": build_environment(device),
    }
    atomic_write_json(args.output_dir / "resolved_config.json", resolved_config)

    records, selected_indices = evaluate(
        args,
        dataset,
        model,
        tokenizer,
        processor,
        device,
        dtype,
    )
    metrics = finalize_results(args, records, selected_indices)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved run: {args.output_dir.resolve()}")

    del model, tokenizer, processor, dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
