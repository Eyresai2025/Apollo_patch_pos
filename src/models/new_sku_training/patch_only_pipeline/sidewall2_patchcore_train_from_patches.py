"""
Sidewall 2 patch-only training

Patch-only PatchCore training for sidewall2.

Edit PATCH_FOLDER and OUT_MODEL_PATH, then run:

    python sidewall2_patchcore_train_from_patches.py

Input  : already-created good/normal patch images
Output : only the .pth PatchCore model
"""

from pathlib import Path

from patchcore_patch_training_core import (
    PatchTrainingConfig,
    train_patchcore_from_patches,
)


# ============================================================================
# USER CONFIGURATION
# ============================================================================

PATCH_FOLDER = Path(
    r"C:/CHANGE_ME/sidewall2/patches_rtor1"
)

OUT_MODEL_PATH = Path(
    r"C:/CHANGE_ME/models/sidewall2_patchcore_model.pth"
)

# Training settings
INPUT_SIZE = 224
IMAGE_BATCH_SIZE = 32

# Keep 0 for the most stable Windows run.
# After it works, try 2 or 4 for speed.
NUM_WORKERS = 0

FEATURE_PATCH_SIZE = 3
CORESET_PERCENTAGE = 0.1
SEED = 0
DEVICE = "auto"

# Search patch images inside subfolders also.
RECURSIVE = True

# Runtime limits
CUDA_VISIBLE_DEVICES = "0"
CPU_THREADS = 1


def main() -> None:
    config = PatchTrainingConfig(
        side_name="sidewall2",
        patch_folder=PATCH_FOLDER,
        out_model_path=OUT_MODEL_PATH,
        input_size=INPUT_SIZE,
        image_batch_size=IMAGE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        feature_patch_size=FEATURE_PATCH_SIZE,
        coreset_percentage=CORESET_PERCENTAGE,
        seed=SEED,
        device=DEVICE,
        recursive=RECURSIVE,
        cuda_visible_devices=CUDA_VISIBLE_DEVICES,
        cpu_threads=CPU_THREADS,
    )

    train_patchcore_from_patches(config)


if __name__ == "__main__":
    main()
