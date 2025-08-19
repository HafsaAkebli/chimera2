
import os
import numpy as np
from PIL import Image, ImageFile
import openslide
from tifffile import imread
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
import heapq
from multiprocessing import cpu_count
import sys
import time
# === Safe loading for large images ===
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def evaluate_patch_metadata(args):
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

        mean_val = patch_np.mean()
        std_val = patch_np.std()
        white_ratio = np.sum(patch_np > 245) / patch_np.size

        if mean_val > 240 or std_val < 10 or white_ratio > 0.9:
            return None

        return (x_wsi, y_wsi, std_val)  # Lightweight: no image returned
    except Exception:
        return None

def extract_patches_in_memory(wsi_path, mask_path, patch_size=512, tissue_threshold=0.95, max_patches=3000):
    """
    Returns: List of top PIL.Image patches selected by std, no disk I/O
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
        #print("⚙️ Evaluating patch metadata in parallel...")
        start_metadata_eval = time.time()
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(evaluate_patch_metadata, args_list))

        end_metadata_eval = time.time()

        valid_coords = [r for r in results if r is not None]
        print(f"✅ Found {len(valid_coords)} valid patches")
        top_coords = heapq.nlargest(max_patches, valid_coords, key=lambda x: x[2])
        print(f"🏁 Selecting top {len(top_coords)} patches by std deviation")
        # Now read patches (small number only)

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

        print(f"✅ Extracted {len(patches)} top-ranked patches")
        print(f"⏱️ Time for metadata evaluation: {end_metadata_eval - start_metadata_eval:.2f} seconds")
        print(f"⏱️ Time for patch extraction: {end_patch_extraction - start_patch_extraction:.2f} seconds")
        return patches

    except Exception as e:
        print(f"❌ Error during patch extraction: {e}")
        return []

def extract_patches_in_memory_2(wsi_path, mask_path, patch_size=512, tissue_threshold=0.95):
    """
    Returns: List of top PIL.Image patches selected by std, no disk I/O
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
        #print("⚙️ Evaluating patch metadata in parallel...")
        start_metadata_eval = time.time()
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            results = list(executor.map(evaluate_patch_metadata, args_list))

        end_metadata_eval = time.time()

        valid_coords = [(r(1), r(2)) for r in results if r is not None]
        print(f"✅ Found {len(valid_coords)} valid patches")
        return valid_coords

    except Exception as e:
        print(f"❌ Error during patch extraction: {e}")
        return []