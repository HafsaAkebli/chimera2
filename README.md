# CHIMERA Task 2 – Multimodal Inference Pipeline

This repository contains the Docker-ready inference code developed by the MITEL-UNIUD team for Task 2 (BCG Response Subtype Prediction) of the CHIMERA Challenge. The goal is to predict Bacillus Calmette–Guérin (BCG) response subtypes (BRS3 vs. BRS1/2) in high-risk non-muscle-invasive bladder cancer (NMIBC) using histopathology and clinical data.

## 📁 Repository Structure

```bash
chimera2/
├── classifier/
│ ├── classifier.py # Fusion MLP for histology + clinical embeddings
│ └── classifier_clinical_only_onehot.py # Clinical-only fallback model
│
├── clinical/
│ └── one_hot_encode.py # Builds one-hot patient clinical vectors
│
├── histology/
│ ├── feature_extraction_uni2.py # Extracts patch features using UNI2-h
│ ├── gat_encoder.py # Aggregates patch features with frozen GAT
│ └── patch_extraction_br.py # Patch selection using Blue-Ratio (cellularity)
│
├── clinical_preproc_meta_T2.json # Metadata for clinical preprocessing
│
├── inference.py # Main inference entrypoint (Grand Challenge)
├── Dockerfile # Container definition
├── do_build.sh # Build Docker image
├── do_test_run.sh # Local test run
├── do_save.sh # Save container as .tar.gz
├── requirements.txt # Dependencies
└── README.md
```
## Inference Pipeline Overview

**1. Patch Extraction:**  
Whole-slide images (WSIs) are tiled into non-overlapping 512×512 patches, retaining only regions with at least 95% tissue content. Each patch is ranked by Blue-Ratio cellularity, and the top 6,000 most cellular patches are selected.  

**2. Feature Extraction (Histology):**  
Selected patches are embedded using the UNI2-h foundation model, a ViT-based encoder pretrained on large-scale histopathology data.  

**3. Graph Aggregation (GAT):**  
Patch embeddings are organized into a cosine KNN graph (k = 5). A frozen 4-layer Graph Attention Network (GAT) with 64 hidden dimensions and 16 attention heads aggregates patch-level information into a single 1,024-dimensional slide representation.  

**4. Clinical Encoding:**  
Structured clinical variables are one-hot encoded according to the precomputed metadata file `clinical_preproc_meta_T2.json`.  

**5. Fusion & Classification:**  
The histology and clinical embeddings are concatenated and passed through a Fusion MLP classifier for BRS subtype prediction.  
If no valid patches are available, a clinical-only fallback model (`classifier_clinical_only_onehot.py`) is executed automatically.  

**6. Output:**  
The final prediction is saved as `brs-probability.json`, containing the predicted probability of BRS3 for the given patient.