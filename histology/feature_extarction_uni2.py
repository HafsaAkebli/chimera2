import torch
import timm
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import multiprocessing
import time
from huggingface_hub import login

# === In-Memory Dataset ===
class InMemoryPatchDataset(Dataset):
    def __init__(self, image_list, transform):
        self.image_list = image_list
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = self.image_list[idx]
        img = img.convert("RGB") if not isinstance(img, torch.Tensor) else img
        return self.transform(img)


# === Model + Transform Loader ===
def load_uni2_model(uni2_path):
    login(token="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")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timm_kwargs = {
        'model_name': 'vit_giant_patch14_224',
        'img_size': 224,
        'patch_size': 14,
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5,
        'embed_dim': 1536,
        'mlp_ratio': 2.66667 * 2,
        'num_classes': 0,
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked,
        'act_layer': torch.nn.SiLU,
        'reg_tokens': 8,
        'dynamic_img_size': True
    }

    model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
    model = model.to(device).eval()

    # ✅ Use exact same transform as in training
    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)


    return model, transform, device


# === In-Memory Feature Extraction ===
def extract_features_from_images(image_list, uni2_path, batch_size=370):
    if len(image_list) == 0:
        print("❌ No patches provided for feature extraction.")
        return np.empty((0, 1536), dtype=np.float32)

    print(f"🔍 Starting feature extraction for {len(image_list)} patches...")

    model, transform, device = load_uni2_model(uni2_path)
    dataset = InMemoryPatchDataset(image_list, transform)

    
    cpu_cores = multiprocessing.cpu_count()
    #num_workers = min(4, cpu_cores - 1) 
    num_workers = max(1, cpu_cores - 1)
    print(f"🧠 Using {num_workers} CPU workers | 🧺 Batch size: {batch_size}")

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    all_features = []
    start = time.time()
    with torch.inference_mode():
        for batch in dataloader:
            batch = batch.to(device)
            features = model(batch).cpu().numpy()
            all_features.append(features)

    all_features = np.vstack(all_features)
    print(f"⏱️ Feature extraction time: {time.time() - start:.2f} seconds")
    del model
    torch.cuda.empty_cache()
    
    print(f"✅ Extracted feature matrix shape: {all_features.shape}")
    return all_features
