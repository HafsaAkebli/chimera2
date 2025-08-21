"""
The following is a simple example algorithm.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torchvision
torchvision.disable_beta_transforms_warning()
from pathlib import Path
import json
from glob import glob
import numpy as np
import torch
from glob import glob
import pyvips
from PIL import Image
# === Safe loading for large images ===
Image.MAX_IMAGE_PIXELS = None

from classifier.classifier_clinical_only_onehot import predict_probability_clinical_only
from classifier.classifier_new import predict_probability
from histology.feature_extraction_uni2_1 import extract_features_from_images
from histology.patch_extraction_br import extract_patches_by_cellularity
from histology.mean_mil import mean_mil_embed
from clinical.one_hot_encode import encode_patient

print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())


INPUT_PATH = Path("/mnt/dmif-nas/MITEL/hafsa/chimera_bcg/Task2/input10")
#OUTPUT_PATH = Path("/output")
OUTPUT_PATH = Path("/mnt/dmif-nas/MITEL/hafsa/chimera_bcg/Task2/model3/output")

MODEL_PATH = Path("/mnt/dmif-nas/MITEL/hafsa/chimera_bcg/Task2/model3")

def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    # Lookup the handler for this particular set of sockets (i.e. the interface)
    handler = {
        (
            "bladder-cancer-tissue-biopsy-whole-slide-image",
            "chimera-clinical-data-of-bladder-cancer-patients",
            "tissue-mask",

        ): interface_0_handler,
    }[interface_key]

    # Call the handler
    return handler()


def interface_0_handler():
    # Read the input
    input_tissue_mask = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/tissue-mask",
        max_size=1024,
    )
    # Use thumbnail loading for large WSI to avoid memory issues
    input_bladder_cancer_tissue_biopsy_whole_slide_image = load_image_file_as_thumbnail(
        location=INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi",
        max_size=1024,
    )
    input_chimera_clinical_data_of_bladder_cancer_patients = load_json_file(
        location=INPUT_PATH / "chimera-clinical-data-of-bladder-cancer-patients.json",
    )
    _show_torch_cuda_info()

    print(f"Clinical data keys: {list(input_chimera_clinical_data_of_bladder_cancer_patients.keys()) if input_chimera_clinical_data_of_bladder_cancer_patients else 'None'}")

    #handle both mha and tif 
    wsi_files = glob(str(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi" / "*.tif")) + \
            glob(str(INPUT_PATH / "images/bladder-cancer-tissue-biopsy-wsi" / "*.mha"))
    
    mask_files = glob(str(INPUT_PATH / "images/tissue-mask" / "*.tif")) + \
             glob(str(INPUT_PATH / "images/tissue-mask" / "*.mha"))

    if not wsi_files or not mask_files:
        raise FileNotFoundError("❌ No WSI or mask file found in the input folder.")
    
    wsi_path = wsi_files[0]  # The one WSI file
    mask_path = mask_files[0]  # The one mask file

    uuid = Path(wsi_path).stem

    print(f"📂 UUID: {uuid}")

    # === Run Patch Extraction + Feature Extraction ===
    #patch_list = extract_patches_in_memory_2(wsi_path, mask_path)
    patch_list = extract_patches_by_cellularity(wsi_path, mask_path)
    print(f"✅ {len(patch_list)} patches extracted for {uuid}")

    if not patch_list:
        print("⚠️ No patches extracted. Running clinical-only fallback model...")

        META_PATH = MODEL_PATH / "clinical/clinical_preproc_meta_T2.json"
        print("\n🧮 Building one-hot clinical vector...")
        clinical_embedding, clinical_cols = encode_patient(
            patient_data=input_chimera_clinical_data_of_bladder_cancer_patients,
            meta_path=str(META_PATH),
        )
        print(f"   ➤ Clinical vector shape: {clinical_embedding.shape}")
        print(f"   ➤ First 5 cols: {clinical_cols[:5]}")
        print(f"   ➤ First 5 values: {clinical_embedding[0, :5]}")

        Classifier_Clinical_Only_PATH = MODEL_PATH / "classifier/clinical_only_classifier.pth"
        
        output_brs_binary_classification = predict_probability_clinical_only(
        clinical_embedding,
        model_path=Classifier_Clinical_Only_PATH)
        
        del clinical_embedding
        torch.cuda.empty_cache()
        print(f"✅ Prediction output: {output_brs_binary_classification}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        
        # Save your output
        write_json_file(
            location=OUTPUT_PATH / "brs-probability.json",
            content=output_brs_binary_classification,
        )
        
        print("\n💾 Prediction saved to:", OUTPUT_PATH / "brs-probability.json")
        print("🎉 Inference completed successfully.")

        return 0
    
    else:

        #UNI2 path
        UNI2_MODEL_PATH = MODEL_PATH / "uni2/pytorch_model.bin"
        features = extract_features_from_images(patch_list, UNI2_MODEL_PATH)

        # ✅ Check if feature extraction succeeded
        if features is None or features.shape[0] == 0:
            print("❌ Feature extraction failed or returned empty feature matrix.")
            raise ValueError("Feature extraction failed. Cannot proceed to GAT.")
        else:
            print(f"✅ Extracted {features.shape[0]} features of dimension {features.shape[1]}")
        
        mean_pooled_vector = features.mean(axis=0).astype(np.float32)
        print("📊 First 5 elements of the mean pooled feature vector:")
        print(mean_pooled_vector[:10])

        MEAN_MIL_PATH = MODEL_PATH / "meanmil/meanMIL_1536_fixed.pt"
        print("\n📊 Starting patient-level embedding using Mean-MIL...")
        print(f"   ➤ MeanMIL model path: {MEAN_MIL_PATH}")

        features = features.astype(np.float32, copy=False)
        histology_embedding = mean_mil_embed(features, str(MEAN_MIL_PATH)) 
        
        print("\n📊 First 10 elements of the histology embedding:")
        print(histology_embedding[:10])

        del features
        torch.cuda.empty_cache()

        META_PATH = MODEL_PATH / "clinical/clinical_preproc_meta_T2.json"
        print("\n🧮 Building one-hot clinical vector...")
        clinical_embedding, clinical_cols = encode_patient(
            patient_data=input_chimera_clinical_data_of_bladder_cancer_patients,
            meta_path=str(META_PATH),
        )
        print(f"   ➤ Clinical vector shape: {clinical_embedding.shape}")
        print(f"   ➤ First 5 cols: {clinical_cols[:5]}")
        print(f"   ➤ First 5 values: {clinical_embedding[0, :5]}")


        Classifier_PATH = MODEL_PATH / "classifier/fusionMLP.pth"

        print("\n🔮 Running final BRS classifier prediction...")
        print(f"   ➤ Classifier path: {Classifier_PATH}")

        output_brs_binary_classification = predict_probability(
        histology_embedding,
        clinical_embedding,
        model_path=Classifier_PATH)

        print(f"✅ Prediction output: {output_brs_binary_classification}")
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        # Save your output
        write_json_file(
            location=OUTPUT_PATH / "brs-probability.json",
            content=output_brs_binary_classification,
        )
        
        print("\n💾 Prediction saved to:", OUTPUT_PATH / "brs-probability.json")
        print("🎉 Inference completed successfully.")
        return 0


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(
        location=INPUT_PATH / "inputs.json",
    )
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())



def write_json_file(*, location, content):
    # Writes a json file
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)

    
def load_image_file_as_thumbnail(*, location, max_size=1024):
    """
    Load image as a thumbnail for memory-efficient processing of WSIs
    This is recommended for actual whole slide images
    Returns the PyVips image object directly for memory efficiency
    """
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
        + glob(str(location / "*.mrxs"))
        + glob(str(location / "*.svs"))
        + glob(str(location / "*.ndpi"))
    )
    
    if not input_files:
        raise FileNotFoundError(f"No compatible image files found in {location}")
    
    file_path = input_files[0]
    print(f"Loading pathology image as thumbnail using PyVips: {file_path}")
    
    # Load image with PyVips
    image = pyvips.Image.new_from_file(file_path)
    
    # Calculate downsampling factor to fit within max_size
    scale_factor = min(max_size / image.width, max_size / image.height)
    if scale_factor < 1.0:
        print(f"Downsampling image by factor {scale_factor:.3f} (from {image.width}x{image.height} to {int(image.width*scale_factor)}x{int(image.height*scale_factor)})")
        image = image.resize(scale_factor)
    else:
        print(f"Image size {image.width}x{image.height} is within max_size={max_size}, no downsampling needed")
    
    # Return the PyVips image object directly (much more memory efficient)
    return image


if __name__ == "__main__":
    raise SystemExit(run())
