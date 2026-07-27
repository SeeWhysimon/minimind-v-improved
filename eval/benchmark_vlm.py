#!/usr/bin/env python3
"""
MiniMind-V inference efficiency benchmark.

Metrics:
1. Source visual-token count from the vision encoder.
2. Visual-token count delivered to the LLM after the visual projector.
3. Time to first generated token (TTFT).
4. Peak CUDA allocated/reserved memory.

The benchmark intentionally excludes image loading and CPU preprocessing from
TTFT. It uses greedy decoding and disables EOS stopping so every measured run
generates the same number of tokens.
"""

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.model_vlm import MiniMindVLM, VLMConfig


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MIB = 1024**2


def parse_args():
    default_output_dir = Path("/data/minimind-v/out/eval_runs")
    parser = argparse.ArgumentParser(
        description="Measure MiniMind-V visual tokens, TTFT, and peak GPU memory."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/minimind-v/out/sft_vlm_768.pth"),
        help="MiniMind-V native PyTorch checkpoint.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "model",
        help="Directory containing tokenizer.json and tokenizer_config.json.",
    )
    parser.add_argument(
        "--vision-model",
        type=Path,
        default=Path("/data/minimind-v/model/siglip2-base-p32-256-ve"),
        help="Local SigLIP2 model and image-processor directory.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=REPO_ROOT / "dataset/eval_images/image-01-golden-dog-balloons.jpg",
        help="One image or a directory of images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. The default is a timestamped file under "
        f"{default_output_dir}.",
    )
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument(
        "--image-token-len",
        type=int,
        default=64,
        help="Visual placeholder count expected by the LLM.",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt",
        type=str,
        default="<image>\n请描述这张图中的主要物体和场景。",
    )
    parser.add_argument("--open-thinking", action="store_true")
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        args.output = default_output_dir / (
            f"{args.checkpoint.stem}-efficiency-{timestamp}.json"
        )
    return args


def validate_args(args):
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
    if not args.image.exists():
        raise FileNotFoundError(f"Image path not found: {args.image}")
    if args.output.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing result: {args.output}"
        )
    if not args.device.startswith("cuda"):
        raise ValueError("Peak GPU-memory measurement requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current PyTorch environment.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if args.image_token_len < 1:
        raise ValueError("--image-token-len must be at least 1.")
    if args.prompt.count("<image>") != 1:
        raise ValueError("--prompt must contain exactly one <image> placeholder.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA device does not support bfloat16.")


def resolve_images(path):
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {path}")
        return [path.resolve()]

    images = sorted(
        item.resolve()
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in: {path}")
    return images


def set_deterministic_seed(seed):
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
    state_dict = {
        name: value
        for name, value in state_dict.items()
        if "mask" not in name
    }
    incompatible = model.load_state_dict(state_dict, strict=False)

    model = model.eval().to(device=device, dtype=dtype)
    return model, tokenizer, model.processor, {
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }


def move_processor_output(processor_output, device, dtype):
    return {
        name: tensor.to(
            device=device,
            dtype=dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in processor_output.items()
    }


def prepare_sample(image_path, model, tokenizer, processor, args, device, dtype):
    with Image.open(image_path) as source_image:
        image = source_image.convert("RGB")
        processor_output = MiniMindVLM.image2tensor(image, processor)

    pixel_values = move_processor_output(processor_output, device, dtype)
    prompt = args.prompt.replace(
        "<image>",
        model.config.image_special_token * model.config.image_token_len,
    )
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    return inputs, pixel_values


class VisualTokenProbe:
    def __init__(self, model):
        self.source_shapes = []
        self.projected_shapes = []
        self.handles = [
            model.vision_encoder.register_forward_hook(self._vision_hook),
            model.vision_proj.register_forward_hook(self._projector_hook),
        ]

    def _vision_hook(self, module, inputs, output):
        hidden_state = output.last_hidden_state
        self.source_shapes.append(tuple(hidden_state.shape))

    def _projector_hook(self, module, inputs, output):
        self.projected_shapes.append(tuple(output.shape))

    def reset(self):
        self.source_shapes.clear()
        self.projected_shapes.clear()

    def result(self):
        if not self.source_shapes:
            raise RuntimeError("The vision-encoder hook did not receive any output.")
        if not self.projected_shapes:
            raise RuntimeError("The visual-projector hook did not receive any output.")

        source_tokens = {shape[-2] for shape in self.source_shapes}
        projected_tokens = {shape[-2] for shape in self.projected_shapes}
        if len(source_tokens) != 1:
            raise RuntimeError(
                f"Inconsistent source visual-token counts: {sorted(source_tokens)}"
            )
        if len(projected_tokens) != 1:
            raise RuntimeError(
                "Inconsistent projected visual-token counts: "
                f"{sorted(projected_tokens)}"
            )
        return {
            "source_visual_tokens_per_image": source_tokens.pop(),
            "llm_visual_tokens_per_image": projected_tokens.pop(),
            "vision_encoder_output_shape": list(self.source_shapes[-1]),
            "vision_projector_output_shape": list(self.projected_shapes[-1]),
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FirstTokenTimer:
    """
    Compatible with MiniMindForCausalLM.generate().

    MiniMind first sends the complete prompt to streamer.put(), then sends one
    generated token per call. The second put() therefore marks the first output
    token.
    """

    def __init__(self, device):
        self.device = device
        self.put_count = 0
        self.first_token_at = None

    def put(self, value):
        self.put_count += 1
        if self.put_count == 2:
            torch.cuda.synchronize(self.device)
            self.first_token_at = time.perf_counter()

    def end(self):
        return None


def run_generation(model, inputs, pixel_values, args, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    streamer = FirstTokenTimer(device)
    started_at = time.perf_counter()
    generated_ids = model.generate(
        inputs=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        eos_token_id=None,
        use_cache=True,
        streamer=streamer,
        pixel_values=pixel_values,
    )
    torch.cuda.synchronize(device)
    finished_at = time.perf_counter()

    if streamer.first_token_at is None:
        raise RuntimeError("No generated token reached the TTFT streamer.")

    prompt_tokens = inputs["input_ids"].shape[-1]
    generated_tokens = generated_ids.shape[-1] - prompt_tokens
    ttft_seconds = streamer.first_token_at - started_at
    total_seconds = finished_at - started_at
    decode_seconds = total_seconds - ttft_seconds
    decode_tokens = max(generated_tokens - 1, 0)
    decode_tokens_per_second = (
        decode_tokens / decode_seconds
        if decode_tokens > 0 and decode_seconds > 0
        else None
    )

    result = {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "ttft_ms": ttft_seconds * 1000,
        "end_to_end_ms": total_seconds * 1000,
        "decode_tokens_per_second": decode_tokens_per_second,
        "baseline_allocated_mib": baseline_allocated / MIB,
        "baseline_reserved_mib": baseline_reserved / MIB,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
        "peak_allocated_delta_mib": (
            torch.cuda.max_memory_allocated(device) - baseline_allocated
        )
        / MIB,
        "peak_reserved_delta_mib": (
            torch.cuda.max_memory_reserved(device) - baseline_reserved
        )
        / MIB,
    }
    del generated_ids
    return result


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def distribution(values):
    return {
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def summarize_iterations(iterations):
    ttft = [item["ttft_ms"] for item in iterations]
    end_to_end = [item["end_to_end_ms"] for item in iterations]
    decode_speed = [
        item["decode_tokens_per_second"]
        for item in iterations
        if item["decode_tokens_per_second"] is not None
    ]
    summary = {
        "ttft_ms": distribution(ttft),
        "end_to_end_ms": distribution(end_to_end),
        "peak_allocated_mib": max(
            item["peak_allocated_mib"] for item in iterations
        ),
        "peak_reserved_mib": max(
            item["peak_reserved_mib"] for item in iterations
        ),
        "peak_allocated_delta_mib": max(
            item["peak_allocated_delta_mib"] for item in iterations
        ),
        "peak_reserved_delta_mib": max(
            item["peak_reserved_delta_mib"] for item in iterations
        ),
    }
    if decode_speed:
        summary["decode_tokens_per_second"] = distribution(decode_speed)
    return summary


def benchmark_image(
    image_path,
    model,
    tokenizer,
    processor,
    probe,
    args,
    device,
    dtype,
):
    inputs, pixel_values = prepare_sample(
        image_path,
        model,
        tokenizer,
        processor,
        args,
        device,
        dtype,
    )
    probe.reset()

    for _ in range(args.warmup):
        run_generation(model, inputs, pixel_values, args, device)

    iterations = []
    for _ in range(args.repeat):
        iterations.append(
            run_generation(model, inputs, pixel_values, args, device)
        )

    result = {
        "image": str(image_path),
        "visual_tokens": probe.result(),
        "summary": summarize_iterations(iterations),
        "iterations": iterations,
    }
    del inputs, pixel_values
    gc.collect()
    return result


def build_environment(device):
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": properties.name,
        "gpu_total_memory_mib": properties.total_memory / MIB,
    }


def aggregate_results(image_results):
    iterations = [
        iteration
        for image_result in image_results
        for iteration in image_result["iterations"]
    ]
    visual_pairs = {
        (
            result["visual_tokens"]["source_visual_tokens_per_image"],
            result["visual_tokens"]["llm_visual_tokens_per_image"],
        )
        for result in image_results
    }
    if len(visual_pairs) != 1:
        raise RuntimeError(
            f"Inconsistent visual-token counts across images: {sorted(visual_pairs)}"
        )

    source_tokens, llm_tokens = visual_pairs.pop()
    return {
        "image_count": len(image_results),
        "measured_iteration_count": len(iterations),
        "source_visual_tokens_per_image": source_tokens,
        "llm_visual_tokens_per_image": llm_tokens,
        **summarize_iterations(iterations),
    }


def write_json_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    validate_args(args)
    images = resolve_images(args.image)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = resolve_dtype(args.dtype)
    set_deterministic_seed(args.seed)

    model, tokenizer, processor, checkpoint_load = load_model(
        args,
        device,
        dtype,
    )
    probe = VisualTokenProbe(model)

    image_results = []
    for image_path in images:
        print(f"Benchmarking: {image_path}")
        image_results.append(
            benchmark_image(
                image_path,
                model,
                tokenizer,
                processor,
                probe,
                args,
                device,
                dtype,
            )
        )
    probe.close()

    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "vision_model_path": str(args.vision_model.resolve()),
        "checkpoint_load": checkpoint_load,
        "model_config": {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": args.use_moe,
            "image_token_len": args.image_token_len,
            "dtype": args.dtype,
        },
        "benchmark_config": {
            "image_input": str(args.image.resolve()),
            "prompt": args.prompt,
            "open_thinking": args.open_thinking,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "decoding": "greedy",
            "eos_stopping": False,
            "ttft_scope": (
                "vision encoder + visual projector + LLM prefill + "
                "first-token selection; excludes image loading and CPU preprocessing"
            ),
        },
        "environment": build_environment(device),
        "aggregate": aggregate_results(image_results),
        "images": image_results,
    }
    write_json_atomic(args.output, payload)

    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    print(f"Saved result: {args.output.resolve()}")


if __name__ == "__main__":
    main()
