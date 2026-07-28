#!/usr/bin/env python3
"""Fine-tune MiniMind-V on converted TextVQA data."""

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset.lm_dataset import VLMDataset
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.trainer_utils import vlm_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune MiniMind-V on TextVQA."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "/data/minimind-v/textvqa/processed/"
            "textvqa_train_sft.parquet"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/data/minimind-v/out/sft_vlm_768.pth"),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=REPO_ROOT / "model",
    )
    parser.add_argument(
        "--vision-model",
        type=Path,
        default=Path(
            "/data/minimind-v/model/siglip2-base-p32-256-ve"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/minimind-v/out/textvqa_runs"),
    )
    parser.add_argument("--run-name", default="textvqa_sft")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=192)
    parser.add_argument("--image-token-len", type=int, default=64)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument(
        "--freeze-llm",
        type=int,
        choices=(0, 1, 2),
        default=1,
        help="0: all non-vision parameters; "
        "1: projector plus first/last LLM layers; "
        "2: projector only.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16"),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args):
    for name, path in {
        "data": args.data_path,
        "checkpoint": args.checkpoint,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} file not found: {path}")

    for name, path in {
        "tokenizer": args.tokenizer_path,
        "vision model": args.vision_model,
    }.items():
        if not path.is_dir():
            raise NotADirectoryError(f"{name} directory not found: {path}")

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be positive.")
    if args.max_seq_len < args.image_token_len + 32:
        raise ValueError("--max-seq-len is too small.")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive.")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1).")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The current GPU does not support bfloat16.")


def initialize_runtime(seed):
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank, rank, world_size = 0, 0, 1

    torch.cuda.set_device(local_rank)
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    return distributed, local_rank, rank, world_size


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer_path),
        local_files_only=True,
    )
    config = VLMConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_seq_len=args.max_seq_len,
        use_moe=args.use_moe,
        image_token_len=args.image_token_len,
    )
    model = MiniMindVLM(
        config,
        vision_model_path=str(args.vision_model),
    )
    if model.vision_encoder is None or model.processor is None:
        raise RuntimeError(
            f"Failed to load vision model: {args.vision_model}"
        )

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    clean_state_dict = {}
    for name, value in state_dict.items():
        if "mask" in name:
            continue
        while name.startswith(("module.", "_orig_mod.")):
            name = name.removeprefix("module.")
            name = name.removeprefix("_orig_mod.")
        clean_state_dict[name] = value

    incompatible = model.load_state_dict(clean_state_dict, strict=False)
    nonvision_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("vision_encoder.")
    ]
    if nonvision_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint and model configuration do not match.\n"
            f"Non-vision missing keys: {nonvision_missing}\n"
            f"Unexpected keys: {list(incompatible.unexpected_keys)}"
        )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "vision_proj" in name:
            parameter.requires_grad = True

    if args.freeze_llm == 0:
        for name, parameter in model.named_parameters():
            if not name.startswith("vision_encoder."):
                parameter.requires_grad = True
    elif args.freeze_llm == 1:
        last_layer = config.num_hidden_layers - 1
        for name, parameter in model.model.named_parameters():
            if (
                "layers.0." in name
                or f"layers.{last_layer}." in name
            ):
                parameter.requires_grad = True

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters were selected.")

    return (
        model.to(device),
        tokenizer,
        config,
        trainable_parameters,
    )


def learning_rate(step, total_steps, warmup_steps, peak):
    if step < warmup_steps:
        return peak * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(
        total_steps - warmup_steps,
        1,
    )
    return peak * (
        0.1 + 0.45 * (1 + math.cos(math.pi * progress))
    )


def move_images(images, device):
    if isinstance(images, dict):
        return {
            name: value.to(device, non_blocking=True)
            for name, value in images.items()
        }
    return images.to(device, non_blocking=True)


def save_model(model, path):
    raw_model = model.module if isinstance(model, DDP) else model
    weights = {
        name: value.half().cpu()
        for name, value in raw_model.state_dict().items()
        if not name.startswith("vision_encoder.")
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(weights, temporary_path)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    validate_args(args)
    distributed, local_rank, rank, world_size = initialize_runtime(
        args.seed
    )
    device = torch.device(f"cuda:{local_rank}")

    run_dir = args.output_dir / args.run_name
    directory_exists = torch.tensor(
        [int(run_dir.exists()) if rank == 0 else 0],
        device=device,
    )
    if distributed:
        dist.broadcast(directory_exists, src=0)
    if directory_exists.item():
        raise FileExistsError(
            f"Run directory already exists: {run_dir}"
        )
    if rank == 0:
        run_dir.mkdir(parents=True)
    if distributed:
        dist.barrier()

    model, tokenizer, config, trainable = load_model(args, device)
    dataset = VLMDataset(
        str(args.data_path),
        tokenizer,
        preprocess=model.processor,
        max_length=args.max_seq_len,
        image_special_token=config.image_special_token,
        image_token_len=config.image_token_len,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=vlm_collate_fn,
    )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_fp16 = args.dtype == "float16"
    amp_dtype = torch.float16 if use_fp16 else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    steps_per_epoch = math.ceil(
        len(loader) / args.accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    if rank == 0:
        resolved_config = vars(args).copy()
        resolved_config.update(
            {
                "data_path": str(args.data_path.resolve()),
                "checkpoint": str(args.checkpoint.resolve()),
                "tokenizer_path": str(args.tokenizer_path.resolve()),
                "vision_model": str(args.vision_model.resolve()),
                "output_dir": str(args.output_dir.resolve()),
                "world_size": world_size,
                "dataset_rows": len(dataset),
                "effective_batch_size": (
                    args.batch_size
                    * world_size
                    * args.accumulation_steps
                ),
                "trainable_parameters": sum(
                    parameter.numel() for parameter in trainable
                ),
            }
        )
        (run_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"rows={len(dataset)}, world_size={world_size}, "
            f"effective_batch_size="
            f"{resolved_config['effective_batch_size']}, "
            f"trainable_parameters="
            f"{resolved_config['trainable_parameters']:,}"
        )

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    log_path = run_dir / "train_log.jsonl"

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()

        for batch_index, (input_ids, labels, images) in enumerate(loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            images = move_images(images, device)

            group_start = (
                batch_index // args.accumulation_steps
            ) * args.accumulation_steps
            group_size = min(
                args.accumulation_steps,
                len(loader) - group_start,
            )
            update = (
                (batch_index + 1) % args.accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            sync_context = (
                nullcontext()
                if update or not isinstance(model, DDP)
                else model.no_sync()
            )

            with sync_context:
                with torch.autocast("cuda", dtype=amp_dtype):
                    output = model(
                        input_ids,
                        labels=labels,
                        pixel_values=images,
                    )
                    loss = output.loss
                    if getattr(output, "aux_loss", None) is not None:
                        loss = loss + output.aux_loss
                scaler.scale(loss / group_size).backward()

            if not update:
                continue

            current_lr = learning_rate(
                global_step,
                total_steps,
                warmup_steps,
                args.learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = current_lr

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if rank == 0 and (
                global_step % args.log_interval == 0
                or batch_index + 1 == len(loader)
            ):
                record = {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "loss": loss.item(),
                    "learning_rate": current_lr,
                }
                print(json.dumps(record, ensure_ascii=False))
                with log_path.open("a", encoding="utf-8") as log_file:
                    log_file.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )

    if distributed:
        dist.barrier()
    if rank == 0:
        suffix = "_moe" if args.use_moe else ""
        output_path = run_dir / (
            f"textvqa_sft_{args.hidden_size}{suffix}.pth"
        )
        save_model(model, output_path)
        print(f"Saved model: {output_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
