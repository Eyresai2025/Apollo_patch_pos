"""PatchCore memory-bank scorer used by threshold generation and live inference.

The five-side runtime keeps one memory bank per SKU/view but shares the heavy
WideResNet feature extractor per device.  This avoids loading five duplicate
backbones when all tyre views are enabled.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import cv2  # type: ignore
import numpy as np
import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image  # type: ignore
from torchvision import models, transforms  # type: ignore

from .config import (
    FEATURE_PATCH_SIZE,
    FEATURE_PATCH_STRIDE,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    MEMORY_BANK_CHUNK_SIZE,
)


_SHARED_BACKBONES: dict[str, nn.Module] = {}
_SHARED_TRANSFORMS: dict[str, object] = {}
_SHARED_LOCK = threading.RLock()
_INFERENCE_LOCKS: dict[str, threading.RLock] = {}


def load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _build_transform():
    return transforms.Compose(
        [
            transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _shared_backbone(device: torch.device) -> nn.Module:
    key = str(device)
    with _SHARED_LOCK:
        model = _SHARED_BACKBONES.get(key)
        if model is not None:
            return model

        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        model = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        ).to(device).eval()

        for parameter in model.parameters():
            parameter.requires_grad = False

        _SHARED_BACKBONES[key] = model
        _SHARED_TRANSFORMS[key] = _build_transform()
        _INFERENCE_LOCKS[key] = threading.RLock()
        return model


def _array_to_pil(image: np.ndarray) -> Image.Image:
    if image is None:
        raise ValueError("Patch image is None")

    if image.ndim == 2:
        return Image.fromarray(image)

    if image.ndim == 3 and image.shape[2] == 1:
        return Image.fromarray(image[:, :, 0])

    if image.ndim == 3 and image.shape[2] == 3:
        array = image
        if array.dtype == np.uint16:
            array = ((array.astype(np.uint32) + 128) // 257).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))

    if image.ndim == 3 and image.shape[2] == 4:
        array = image
        if array.dtype == np.uint16:
            array = ((array.astype(np.uint32) + 128) // 257).astype(np.uint8)
        return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA))

    raise ValueError(f"Unsupported patch array shape: {image.shape}")


class PatchCoreScorer:
    """Score images against one saved normalized PatchCore memory bank."""

    def __init__(self, model_path: Path, device: str | torch.device | None = None):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        requested_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        checkpoint = load_torch_file(self.model_path)
        if not isinstance(checkpoint, dict) or "memory_bank" not in checkpoint:
            raise KeyError("The model file must contain a 'memory_bank' tensor.")

        memory_bank = checkpoint["memory_bank"]
        if not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 2:
            raise ValueError("'memory_bank' must be a 2-D PyTorch tensor.")

        self.memory_bank = F.normalize(
            memory_bank.detach().float(), p=2, dim=1
        ).to(self.device)

        self.feature_extractor = _shared_backbone(self.device)
        key = str(self.device)
        self.transform = _SHARED_TRANSFORMS[key]
        self._inference_lock = _INFERENCE_LOCKS[key]

    def _nearest_memory_distance(self, query_patches: torch.Tensor) -> torch.Tensor:
        best_distance = torch.full(
            (query_patches.shape[0],),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )

        for start in range(0, self.memory_bank.shape[0], MEMORY_BANK_CHUNK_SIZE):
            bank_chunk = self.memory_bank[start : start + MEMORY_BANK_CHUNK_SIZE]
            distances = torch.cdist(query_patches, bank_chunk, p=2)
            best_distance = torch.minimum(best_distance, distances.min(dim=1).values)
        return best_distance

    @torch.inference_mode()
    def _score_pil_batch(self, images: Sequence[Image.Image]) -> list[float]:
        if not images:
            return []

        with self._inference_lock:
            tensors = [self.transform(image.convert("RGB")) for image in images]
            image_batch = torch.stack(tensors, dim=0).to(self.device)
            features = self.feature_extractor(image_batch)
            unfolded = F.unfold(
                features,
                kernel_size=FEATURE_PATCH_SIZE,
                stride=FEATURE_PATCH_STRIDE,
            )
            feature_patches = unfolded.transpose(1, 2).contiguous()
            batch_size, internal_patch_count, feature_dimension = feature_patches.shape

            if feature_dimension != self.memory_bank.shape[1]:
                raise ValueError(
                    "Feature dimension mismatch. "
                    f"Extracted={feature_dimension}, memory_bank={self.memory_bank.shape[1]}."
                )

            query_patches = feature_patches.reshape(-1, feature_dimension)
            query_patches = F.normalize(query_patches.float(), p=2, dim=1)
            nearest_distances = self._nearest_memory_distance(query_patches)
            nearest_distances = nearest_distances.view(batch_size, internal_patch_count)
            scores = nearest_distances.max(dim=1).values
            return [float(value) for value in scores.cpu().tolist()]

    def score_batch(self, image_paths: list[Path]) -> list[float]:
        images: list[Image.Image] = []
        try:
            for image_path in image_paths:
                images.append(Image.open(image_path))
            return self._score_pil_batch(images)
        finally:
            for image in images:
                try:
                    image.close()
                except Exception:
                    pass

    def score_array_batch(self, images: list[np.ndarray]) -> list[float]:
        """Score in-memory patches without writing temporary patch files."""
        pil_images = [_array_to_pil(image) for image in images]
        try:
            return self._score_pil_batch(pil_images)
        finally:
            for image in pil_images:
                try:
                    image.close()
                except Exception:
                    pass
