import glob, cv2, os, os.path as osp
from patchify import patchify
import numpy as np

def patchify_index_grouped(source_path, patch_h, patch_w,
                           step_h=None, step_w=None, cover_edges=False):

    # Output folder
    if osp.isdir(source_path):
        base_out = osp.join(source_path, "patches_rtor1")
    else:
        base_out = osp.join(osp.dirname(source_path), "patches_rtor1")

    os.makedirs(base_out, exist_ok=True)

    # ---- Handle both folder and single image ----
    if osp.isfile(source_path):
        image_files = [source_path]
    else:
        image_files = sorted(
            glob.glob(osp.join(source_path, "*.jpg")) +
            glob.glob(osp.join(source_path, "*.jpeg")) +
            glob.glob(osp.join(source_path, "*.png"))
        )

    if len(image_files) == 0:
        print("❌ No images found!")
        return

    for file_path in image_files:

        print(f"Patching: {file_path}")

        img = cv2.imread(file_path)

        if img is None:
            print(f"⚠️ Could not read image: {file_path}")
            continue

        H, W = img.shape[:2]

        if H < patch_h or W < patch_w:
            print(f"⚠️ Skipping small image (H{H}xW{W})")
            continue

        ch = img.shape[2] if img.ndim == 3 else 1

        step_h = patch_h if step_h is None else step_h
        step_w = patch_w if step_w is None else step_w

        filename_base, ext = osp.splitext(osp.basename(file_path))

        # keep original extension (png stays png)
        ext = ext.lower()

        if not cover_edges:

            patches = patchify(
                img, 
                (patch_h, patch_w, ch), 
                step=(step_h, step_w, ch)
            )

            rows, cols = patches.shape[0], patches.shape[1]

            for i in range(rows):
                for j in range(cols):

                    patch = patches[i, j, 0]

                    out_name = f"{filename_base}__r{i:03d}_c{j:03d}{ext}"
                    out_path = osp.join(base_out, out_name)

                    cv2.imwrite(out_path, patch)

        else:
            # ---- Cover edges fully ----

            i_starts = list(range(0, H - patch_h + 1, step_h))
            j_starts = list(range(0, W - patch_w + 1, step_w))

            if i_starts[-1] != H - patch_h:
                i_starts.append(H - patch_h)

            if j_starts[-1] != W - patch_w:
                j_starts.append(W - patch_w)

            for r, i0 in enumerate(i_starts):
                for c, j0 in enumerate(j_starts):

                    patch = img[i0:i0+patch_h, j0:j0+patch_w]

                    out_name = f"{filename_base}__r{r:03d}_c{c:03d}{ext}"
                    out_path = osp.join(base_out, out_name)

                    cv2.imwrite(out_path, patch)

    print("\n✅ Done! Patches saved in:")
    print(base_out)


# ---------------- RUN ----------------

if __name__ == "__main__":

    source_path = r"D:\ceat_data\20_sw1_crack_fulltyre\test"
    # OR
    # source_path = r"C:\Users\cmrit\Downloads\full tyre\cropped"

    patchify_index_grouped(
        source_path,
        patch_h=448,
        patch_w=448,
        step_h=448,
        step_w=448,
        cover_edges=True
    )