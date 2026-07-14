import os
import time
from datetime import datetime
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ==================== CONFIGURATION ====================
TRAIN_PATH = r"C:\Users\YerriswamyChakala\Downloads\patchcore\Exp1Apollo-glitter\train\good"
OUT_PATH = r"C:\Users\YerriswamyChakala\Downloads\patchcore\G44\tread2.pth"
TIMING_CSV = r"C:\Users\YerriswamyChakala\Downloads\patchcore\G44\timings.csv"
# NOTE: intermediate batch files are no longer written to disk -- patches are kept in
# memory and concatenated once at the end, so BATCH_DIR is no longer needed.

# PatchCore parameters (official configuration)
PATCH_SIZE = 3
LAYERS_TO_EXTRACT = ["layer2", "layer3"]
CORESET_PERCENTAGE = 0.1
INPUT_SIZE = 224
IMG_BATCH_SIZE = 32       # images per GPU forward pass; lower this first if you hit GPU OOM
NUM_WORKERS = min(4, os.cpu_count() or 1)  # if your dataset is small, also try NUM_WORKERS=0 --
                                            # process-spawn startup can cost more than it saves at small scale


# ==================== DATASET (module-level so it can be pickled to worker processes) ====================
class ImageListDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
            return tensor, path, True
        except Exception:
            return torch.zeros(3, INPUT_SIZE, INPUT_SIZE), path, False


# ==================== TIMING HELPERS ====================
def now_s():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def elapsed_str(start):
    secs = time.perf_counter() - start
    return f"{secs:.2f}s"


def main():
    csv_fields = ["timestamp", "stage", "item_start", "item_end", "count", "duration_s", "notes"]
    csv_rows = []

    def csv_log(stage, item_start=None, item_end=None, count=None, duration_s=None, notes=""):
        csv_rows.append([
            now_s(), stage,
            item_start if item_start is not None else "",
            item_end if item_end is not None else "",
            count if count is not None else "",
            f"{duration_s:.4f}" if duration_s is not None else "",
            str(notes)
        ])

    start_all = time.perf_counter()
    start_all_dt = datetime.now()
    print(f"[{now_s()}] Run started")

    # ==================== MODEL & TRANSFORM ====================
    print(f"[{now_s()}] Loading WideResNet-50 (recommended for PatchCore)...")
    model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
    print(f"[{now_s()}] CHECKPOINT: weights loaded")

    feature_extractor = nn.Sequential(
        model.conv1, model.bn1, model.relu, model.maxpool,
        model.layer1, model.layer2, model.layer3
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.eval().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad = False
    print(f"[{now_s()}] CHECKPOINT: model moved to {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    transform = transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[{now_s()}] Loading images...")
    imgs = [os.path.join(TRAIN_PATH, f) for f in os.listdir(TRAIN_PATH)
            if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
    print(f"[{now_s()}] CHECKPOINT: found {len(imgs)} images")

    # ==================== FEATURE EXTRACTION (in-memory, no intermediate disk I/O) ====================
    successful = 0
    processed_count = 0
    patch_chunks = []   # list of CPU tensors; concatenated once at the very end
    total_patch_count = 0

    print("Extracting features from layer2 + layer3 (batched, prefetched, in-memory)...")

    with torch.no_grad():
        dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE).to(device)
        feat_dummy = feature_extractor(dummy)
        print(f"[{now_s()}] CHECKPOINT: warmup forward done, feature shape {feat_dummy.shape}")

    csv_log("start_total", 0, len(imgs), None, 0.0, "script_start")

    dataset = ImageListDataset(imgs, transform)
    loader = DataLoader(
        dataset,
        batch_size=IMG_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=4 if NUM_WORKERS > 0 else None,
    )
    print(f"[{now_s()}] CHECKPOINT: DataLoader created (num_workers={NUM_WORKERS}, batch_size={IMG_BATCH_SIZE})")

    print_every_n_chunks = max(1, 100 // IMG_BATCH_SIZE)

    for chunk_idx, (tensors_batch, paths_batch, valid_batch) in enumerate(loader):
        chunk_start_idx = chunk_idx * IMG_BATCH_SIZE
        do_print = (chunk_idx % print_every_n_chunks == 0)

        if do_print:
            print(f"[{now_s()}] Progress: {processed_count}/{len(imgs)} ({total_patch_count} patches)")

        valid_list = valid_batch.tolist()
        for p, ok in zip(paths_batch, valid_list):
            if not ok:
                print(f"[{now_s()}] Error loading {p}")

        processed_count += len(paths_batch)
        num_valid = sum(valid_list)
        if num_valid == 0:
            continue

        valid_tensors = tensors_batch[valid_batch]
        valid_paths = [p for p, ok in zip(paths_batch, valid_list) if ok]

        try:
            input_batch = valid_tensors.to(device, non_blocking=True)

            chunk_wall_start = time.perf_counter()
            with torch.inference_mode():
                with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    feat = feature_extractor(input_batch)  # [B, C, H, W]
                feat = feat.float()

            B, C, H, W = feat.shape
            Hout = (H - PATCH_SIZE) // PATCH_SIZE + 1
            Wout = (W - PATCH_SIZE) // PATCH_SIZE + 1

            patches = feat.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B * Hout * Wout, C * PATCH_SIZE * PATCH_SIZE)

            # Move to CPU immediately to keep GPU memory free for the next chunk --
            # no disk write, just an in-memory handoff.
            patch_chunks.append(patches.cpu())
            total_patch_count += patches.shape[0]
            successful += len(valid_paths)

            if do_print:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                chunk_elapsed = time.perf_counter() - chunk_wall_start
                print(f"[{now_s()}] Chunk {chunk_idx}: {B} images, "
                      f"chunk_time={chunk_elapsed:.3f}s, new_patches={patches.shape[0]}")
                csv_log("image_progress", chunk_start_idx, chunk_start_idx + B - 1, B, chunk_elapsed, "")

        except Exception as e:
            print(f"[{now_s()}] Error processing chunk starting at {chunk_start_idx}: {str(e)[:200]}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[{now_s()}] Processed: {successful}/{len(imgs)}")
    print(f"[{now_s()}] Total patches extracted: {total_patch_count}")
    csv_log("after_extraction", 0, len(imgs) - 1, total_patch_count, time.perf_counter() - start_all, "extraction_done")

    # ==================== BUILD MEMORY BANK (already in memory -- just concatenate + normalize once) ====================
    print(f"\n[{now_s()}] Building memory bank...")
    mem_start = time.perf_counter()
    if patch_chunks:
        all_patches = torch.cat(patch_chunks, dim=0)
        all_patches = F.normalize(all_patches, p=2, dim=1)
    else:
        all_patches = torch.empty((0,))
    mem_elapsed = time.perf_counter() - mem_start
    print(f"[{now_s()}] Total patches assembled: {all_patches.shape}, took {mem_elapsed:.2f}s")
    csv_log("build_memory_bank", 0, 0, all_patches.shape if patch_chunks else 0, mem_elapsed, "in_memory_concat")

    # Coreset subsampling (random sampling substitute for greedy coreset)
    if all_patches.numel() == 0:
        print(f"[{now_s()}] Warning: no patches found to build memory bank.")
        memory_bank = all_patches
    else:
        num_keep = max(1, int(len(all_patches) * CORESET_PERCENTAGE))
        indices = torch.randperm(len(all_patches))[:num_keep]
        memory_bank = all_patches[indices]
        print(f"[{now_s()}] Memory bank size: {memory_bank.shape} patches ({CORESET_PERCENTAGE * 100:.1f}% of total)")
        csv_log("coreset_subsample", 0, 0, memory_bank.shape, 0.0, f"percentage={CORESET_PERCENTAGE}")

    # Final save (memory bank)
    final_save_start = time.perf_counter()
    torch.save({'memory_bank': memory_bank}, OUT_PATH)
    final_save_elapsed = time.perf_counter() - final_save_start
    print(f"[{now_s()}] Final memory bank saved to {OUT_PATH} in {final_save_elapsed:.2f}s")
    csv_log("final_save", 0, 0, memory_bank.shape, final_save_elapsed, f"out_path={OUT_PATH}")

    print(f"\n[{now_s()}] No intermediate batch files were written (in-memory mode) -- nothing to clean up.")
    csv_log("cleanup", 0, 0, 0, 0.0, "no_disk_batches_used")

    end_all = time.perf_counter()
    end_all_dt = datetime.now()
    total_elapsed = end_all - start_all
    print(f"[{now_s()}] Run finished: {end_all_dt}")
    print(f"[{now_s()}] Total run time: {elapsed_str(start_all)} (from {start_all_dt} to {end_all_dt})")
    csv_log("end_total", 0, 0, total_patch_count, total_elapsed, f"run_start={start_all_dt},run_end={end_all_dt}")

    try:
        with open(TIMING_CSV, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(csv_fields)
            writer.writerows(csv_rows)
        print(f"[{now_s()}] Timing CSV written to {TIMING_CSV}")
    except Exception as e:
        print(f"[{now_s()}] Failed writing timing CSV: {e}")

    print(f"[{now_s()}] TRAINING DONE")


if __name__ == "__main__":
    main()