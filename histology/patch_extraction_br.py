import os
import numpy as np
from PIL import Image, ImageFile
import openslide
from tifffile import imread
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
import heapq
import sys
import time
from scipy.ndimage import gaussian_filter

# === Safe loading for large images ===
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Helper Functions ===
def is_mha(path):
    return path.lower().endswith(".mha")

def load_mask(mask_path):
    if is_mha(mask_path):
        mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
        if mask.ndim == 3:
            mask = mask[0]
    else:
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]
    return mask

def load_wsi_dims(wsi_path):
    if is_mha(wsi_path):
        img = sitk.ReadImage(wsi_path)
        arr = sitk.GetArrayFromImage(img)[0]
        return arr.shape[1], arr.shape[0]
    else:
        slide = openslide.OpenSlide(wsi_path)
        dims = slide.dimensions
        slide.close()
        return dims


def read_patch(wsi_path, x, y, patch_size):
    if is_mha(wsi_path):
        img = sitk.GetArrayFromImage(sitk.ReadImage(wsi_path))[0]
        patch_np = img[y:y+patch_size, x:x+patch_size, ...]
        return Image.fromarray(patch_np)
    else:
        slide = openslide.OpenSlide(wsi_path)
        patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
        slide.close()
        return patch

# === Blue Ratio Cellularity Scoring ===
def compute_cellularity_score(patch_np):
    R, G, B = patch_np[..., 0], patch_np[..., 1], patch_np[..., 2]
    denom = (R + G + B).astype(np.float32) + 1e-5
    br_image = 100 * B / denom * (1 / (1 + denom))
    br_smooth = gaussian_filter(br_image, sigma=1.0)
    threshold = np.percentile(br_smooth, 90)
    binary_mask = br_smooth > threshold
    return np.sum(binary_mask) / binary_mask.size


def evaluate_patch_cellularity(args):
    (i, j, stride_y, stride_x, mask_np, scale_x, scale_y,
     wsi_path, wsi_w, wsi_h, patch_size, tissue_threshold) = args

    mask_crop = mask_np[i:i + stride_y, j:j + stride_x]
    if mask_crop.shape != (stride_y, stride_x):
        return None

    tissue_ratio = np.count_nonzero(mask_crop) / mask_crop.size
    if tissue_ratio < tissue_threshold:
        return None

    x_wsi = int(j * scale_x)
    y_wsi = int(i * scale_y)
    if x_wsi + patch_size > wsi_w or y_wsi + patch_size > wsi_h:
        return None

    try:
        patch = read_patch(wsi_path, x_wsi, y_wsi, patch_size)
        patch_np = np.array(patch)

        # Filter out blank or low-contrast patches
        mean_val = patch_np.mean()
        std_val = patch_np.std()
        white_ratio = np.sum(patch_np > 245) / patch_np.size

        if mean_val > 240 or std_val < 10 or white_ratio > 0.9:
            return None

        score = compute_cellularity_score(patch_np)
        return (x_wsi, y_wsi, score)
    except Exception:
        return None

# === Patch Extraction by Cellularity ===
def extract_patches_by_cellularity(wsi_path, mask_path, patch_size=512, tissue_threshold=0.95, max_patches=6000):
    """
    Returns: List of top PIL.Image patches selected by cellularity (Blue Ratio),
    sorted to match training patch order (Y, then X).
    """
    try:
        mask_np = load_mask(mask_path)
        mask_h, mask_w = mask_np.shape

        wsi_w, wsi_h = load_wsi_dims(wsi_path)
        scale_x = wsi_w / mask_w
        scale_y = wsi_h / mask_h

        stride_x = int(patch_size / scale_x)
        stride_y = int(patch_size / scale_y)

        print("🧵 Generating candidate patch coordinates...")
        args_list = [
            (
                i, j, stride_y, stride_x,
                mask_np, scale_x, scale_y,
                wsi_path, wsi_w, wsi_h,
                patch_size, tissue_threshold
            )
            for i in range(0, mask_h - stride_y, stride_y)
            for j in range(0, mask_w - stride_x, stride_x)
        ]

        start_metadata_eval = time.time()

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(evaluate_patch_cellularity, args_list))

        end_metadata_eval = time.time()

        valid_coords = [r for r in results if r is not None]
        print(f"✅ Found {len(valid_coords)} valid patches")
        if not valid_coords:
            return []

        top_coords = heapq.nlargest(max_patches, valid_coords, key=lambda x: x[2])
        print(f"🏁 Selecting top {len(top_coords)} patches by cellularity score")
        print("🧭 First 5 patch coordinates:", top_coords[:5])

        # Now read the actual top patches
        print("🖼️ Extracting top-ranked patches...")
        start_patch_extraction = time.time()

        patches = []
        for x, y, _ in top_coords:
            try:
                patch = read_patch(wsi_path, x, y, patch_size)
                patches.append(patch)
            except Exception:
                continue
        end_patch_extraction = time.time()

        total_bytes = sum(sys.getsizeof(patch.tobytes()) for patch in patches)
        print(f"📦 Estimated memory used by {len(patches)} patches: {total_bytes / (1024 ** 2):.2f} MB")
        print(f"⏱️ Metadata evaluation time: {end_metadata_eval - start_metadata_eval:.2f} s")
        print(f"⏱️ Patch extraction time: {end_patch_extraction - start_patch_extraction:.2f} s")

        return patches

    except Exception as e:
        print(f"❌ Error during patch extraction: {e}")
        return []