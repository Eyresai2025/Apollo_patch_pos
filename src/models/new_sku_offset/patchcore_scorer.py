"""PatchCore memory-bank scorer matching the supplied tread setup pipeline."""

from __future__ import annotations

from pathlib import Path

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.nn.functional as F  # type: ignore
from PIL import Image  # type: ignore
from torchvision import models, transforms  # type: ignore

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
FEATURE_PATCH_SIZE = 3
FEATURE_PATCH_STRIDE = 3
IMAGE_BATCH_SIZE = 16
MEMORY_BANK_CHUNK_SIZE = 10_000


def _load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PatchCoreScorer:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"PatchCore model not found: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = _load_torch_file(self.model_path)
        if not isinstance(checkpoint, dict) or "memory_bank" not in checkpoint:
            raise KeyError("The model file must contain a 'memory_bank' tensor.")

        memory_bank = checkpoint["memory_bank"]
        if not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 2:
            raise ValueError("'memory_bank' must be a 2-D PyTorch tensor.")

        self.memory_bank = F.normalize(
            memory_bank.detach().float(), p=2, dim=1
        ).to(self.device)

        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )
        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        ).to(self.device).eval()
        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _nearest_memory_distance(self, query_patches: torch.Tensor) -> torch.Tensor:
        best = torch.full(
            (query_patches.shape[0],),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )
        for start in range(0, self.memory_bank.shape[0], MEMORY_BANK_CHUNK_SIZE):
            chunk = self.memory_bank[start : start + MEMORY_BANK_CHUNK_SIZE]
            distances = torch.cdist(query_patches, chunk, p=2)
            best = torch.minimum(best, distances.min(dim=1).values)
        return best

    @torch.inference_mode()
    def score_batch(self, image_paths: list[Path]) -> list[float]:
        if not image_paths:
            return []

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self.transform(image.convert("RGB")))

        image_batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.feature_extractor(image_batch)
        unfolded = F.unfold(
            features,
            kernel_size=FEATURE_PATCH_SIZE,
            stride=FEATURE_PATCH_STRIDE,
        )
        patches = unfolded.transpose(1, 2).contiguous()
        batch_size, internal_patch_count, feature_dim = patches.shape

        if feature_dim != self.memory_bank.shape[1]:
            raise ValueError(
                "Feature dimension mismatch. "
                f"Extracted={feature_dim}, memory_bank={self.memory_bank.shape[1]}."
            )

        query = F.normalize(patches.reshape(-1, feature_dim).float(), p=2, dim=1)
        nearest = self._nearest_memory_distance(query)
        scores = nearest.view(batch_size, internal_patch_count).max(dim=1).values
        return [float(value) for value in scores.cpu().tolist()]
