"""
patchcore_patch_training_core.py

Patch-folder -> PatchCore memory-bank model.

Input : folder containing already-created good/normal patch images.
Output: only one model file: {"memory_bank": memory_bank}

No R detection, no raw crop, no resize, no Vit_patch, no preprocessing outputs.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PatchTrainingConfig:
    side_name: str
    patch_folder: Path
    out_model_path: Path
    input_size: int = 224
    image_batch_size: int = 32
    num_workers: int = 0
    feature_patch_size: int = 3
    coreset_percentage: float = 0.1
    seed: int = 0
    device: str = "auto"
    recursive: bool = True
    cuda_visible_devices: str | None = None
    cpu_threads: int = 1


def natural_key(path: Path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)]


def list_patch_images(folder: Path, recursive: bool = True) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Patch folder not found: {folder}")

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    patches = sorted(
        (p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=natural_key,
    )
    if not patches:
        raise RuntimeError(f"No patch images found in: {folder}")
    return patches


class PatchImageDataset(Dataset):
    def __init__(self, paths: list[Path], transform, input_size: int):
        self.paths = paths
        self.transform = transform
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
            return tensor, str(path), True
        except Exception:
            return torch.zeros(3, self.input_size, self.input_size), str(path), False


def resolve_device(device_setting: str) -> torch.device:
    setting = str(device_setting or "auto").strip().lower()
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE='cuda' was requested, but CUDA is not available.")
        return torch.device("cuda")
    if setting == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: auto, cuda, cpu")


def apply_runtime_limits(config: PatchTrainingConfig) -> None:
    if config.cuda_visible_devices is not None and str(config.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_visible_devices)

    cpu_threads = max(1, int(config.cpu_threads))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)

    try:
        torch.set_num_threads(cpu_threads)
    except Exception:
        pass


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_patchcore_from_patches(config: PatchTrainingConfig) -> dict:
    apply_runtime_limits(config)
    start_time = time.perf_counter()

    patch_folder = Path(config.patch_folder)
    out_model_path = Path(config.out_model_path)
    out_model_path.parent.mkdir(parents=True, exist_ok=True)

    patch_paths = list_patch_images(patch_folder, recursive=bool(config.recursive))
    device = resolve_device(config.device)

    print("=" * 78)
    print(f"PATCHCORE PATCH-ONLY TRAINING: {config.side_name}")
    print("=" * 78)
    print(f"Patch folder       : {patch_folder}")
    print(f"Patch count        : {len(patch_paths)}")
    print(f"Output model       : {out_model_path}")
    print(f"Device             : {device}")
    print(f"Input size         : {config.input_size}")
    print(f"Image batch size   : {config.image_batch_size}")
    print(f"Num workers        : {config.num_workers}")
    print(f"Coreset percentage : {config.coreset_percentage}")

    torch.manual_seed(int(config.seed))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("\nLoading WideResNet-50 feature extractor...")
    backbone = models.wide_resnet50_2(
        weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
    )
    feature_extractor = nn.Sequential(
        backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        backbone.layer1, backbone.layer2, backbone.layer3,
    ).to(device).eval()
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = PatchImageDataset(patch_paths, transform=transform, input_size=int(config.input_size))
    loader_kwargs = {
        "batch_size": max(1, int(config.image_batch_size)),
        "shuffle": False,
        "num_workers": max(0, int(config.num_workers)),
        "pin_memory": (device.type == "cuda"),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(dataset, **loader_kwargs)

    print("\nExtracting features and building memory candidates...")
    patch_chunks: list[torch.Tensor] = []
    processed_images = 0
    successful_images = 0
    failed_images: list[str] = []
    total_feature_patches = 0

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, config.input_size, config.input_size).to(device)
        _ = feature_extractor(dummy)

    synchronize_cuda()

    with torch.inference_mode():
        for batch_index, (image_batch, path_batch, valid_batch) in enumerate(loader):
            valid_mask = valid_batch.bool()
            processed_images += len(path_batch)
            successful_images += int(valid_mask.sum().item())

            for path, ok in zip(path_batch, valid_mask.tolist()):
                if not ok:
                    failed_images.append(str(path))

            if int(valid_mask.sum().item()) == 0:
                continue

            valid_images = image_batch[valid_mask].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                features = feature_extractor(valid_images)

            features = features.float()
            unfolded = F.unfold(
                features,
                kernel_size=int(config.feature_patch_size),
                stride=int(config.feature_patch_size),
            )
            feature_patches = unfolded.transpose(1, 2).contiguous()
            batch_size, internal_patch_count, feature_dimension = feature_patches.shape

            feature_patches = feature_patches.reshape(batch_size * internal_patch_count, feature_dimension)
            feature_patches = F.normalize(feature_patches, p=2, dim=1)

            patch_chunks.append(feature_patches.cpu())
            total_feature_patches += int(feature_patches.shape[0])

            if batch_index == 0 or processed_images == len(patch_paths) or batch_index % 10 == 0:
                print(f"Processed {processed_images}/{len(patch_paths)} images "
                      f"({total_feature_patches} feature patches)")

            del valid_images, features, unfolded, feature_patches

    if not patch_chunks:
        raise RuntimeError("No valid patches were processed; model cannot be trained.")

    print("\nConcatenating feature patches...")
    all_patches = torch.cat(patch_chunks, dim=0)

    if not 0.0 < float(config.coreset_percentage) <= 1.0:
        raise ValueError("coreset_percentage must be in the range (0, 1].")

    coreset_size = max(1, int(all_patches.shape[0] * float(config.coreset_percentage)))
    print(f"Selecting random coreset: {coreset_size}/{all_patches.shape[0]} "
          f"({float(config.coreset_percentage) * 100:.2f}%)")

    generator = torch.Generator()
    generator.manual_seed(int(config.seed))
    indices = torch.randperm(all_patches.shape[0], generator=generator)[:coreset_size]
    memory_bank = all_patches[indices].contiguous()

    torch.save({"memory_bank": memory_bank}, out_model_path)
    elapsed = time.perf_counter() - start_time

    summary = {
        "side_name": config.side_name,
        "patch_folder": str(patch_folder),
        "out_model_path": str(out_model_path),
        "patch_image_count": len(patch_paths),
        "successful_patch_image_count": successful_images,
        "failed_patch_image_count": len(failed_images),
        "failed_patch_images": failed_images,
        "feature_patch_count_before_coreset": int(all_patches.shape[0]),
        "memory_bank_patch_count": int(memory_bank.shape[0]),
        "memory_bank_feature_dimension": int(memory_bank.shape[1]),
        "coreset_percentage": float(config.coreset_percentage),
        "input_size": int(config.input_size),
        "feature_patch_size": int(config.feature_patch_size),
        "device": str(device),
        "elapsed_seconds": float(elapsed),
    }

    print("\n" + "=" * 78)
    print(f"TRAINING COMPLETED: {config.side_name}")
    print("=" * 78)
    print(f"Model saved         : {out_model_path}")
    print(f"Memory bank patches : {memory_bank.shape[0]}")
    print(f"Feature dimension   : {memory_bank.shape[1]}")
    print(f"Elapsed             : {elapsed:.2f} sec")
    return summary
