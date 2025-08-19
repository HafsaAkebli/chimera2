from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def build_clinical_text_from_json(patient_data: dict, patient_id: str) -> str:
    # Wrap dict into a one-row DataFrame
    df = pd.DataFrame([patient_data])

    # Replace -1 with "Missing"
    df = df.replace(-1, "Missing")

    # Fill NAs with "Missing"
    df = df.fillna("Missing")

    # Standardize some fields
    if "sex" in df.columns:
        df['sex'] = df['sex'].str.lower().map({"m": "Male", "f": "Female", "male": "Male", "female": "Female"})
    if "smoking" in df.columns:
        df['smoking'] = df['smoking'].str.capitalize()
    if "tumor" in df.columns:
        df['tumor'] = df['tumor'].str.capitalize()
    if "reTUR" in df.columns:
        df['reTUR'] = df['reTUR'].replace({'yes': 'Yes', 'no': 'No'})
    if "LVI" in df.columns:
        df['LVI'] = df['LVI'].replace({'yes': 'Yes', 'no': 'No'})

    # Build the text
    row = df.iloc[0]

    description = f"This patient is diagnosed with bladder cancer. "
    description += f"The patient's age is {int(row['age'])} years. " if row['age'] != "Missing" else "The patient's age is missing. "
    description += f"The biological sex of the patient is {row['sex']}. "

    # smoking
    if row['smoking'] == "Yes":
        description += "The patient has a history of smoking. "
    elif row['smoking'] == "No":
        description += "The patient does not have a history of smoking. "
    elif row['smoking'] == "Missing":
        description += "Smoking history information is missing. "

    # tumor
    if row['tumor'] == "Primary":
        description += "The patient presents with a primary bladder tumor. "
    elif row['tumor'] == "Recurrence":
        description += "The patient presents with a recurrent bladder tumor. "
    elif row['tumor'] == "Missing":
        description += "Tumor status (primary or recurrence) is missing. "

    # stage
    if row['stage'] == "TaHG":
        description += "The clinical stage of the tumor is Ta high-grade, meaning it is confined to the inner lining of the bladder. "
    elif row['stage'] == "T1HG":
        description += "The clinical stage of the tumor is T1 high-grade, indicating invasion into the connective tissue beneath the bladder lining. "
    elif row['stage'] == "T2HG":
        description += "The clinical stage of the tumor is T2 high-grade, meaning the tumor has invaded the bladder muscle. "
    elif row['stage'] == "T3HG":
        description += "The clinical stage of the tumor is T3 high-grade, indicating tumor invasion into the perivesical tissue surrounding the bladder. "
    elif row['stage'] == "Missing":
        description += "The clinical stage of the tumor is missing. "

    # substage
    if row['substage'] == "T1m":
        description += "The bladder cancer substage is T1m, indicating a tumor invasion depth of 0.5 millimeters or less. "
    elif row['substage'] == "T1e":
        description += "The bladder cancer substage is T1e, indicating a tumor invasion depth greater than 0.5 millimeters. "
    elif row['substage'] == "Missing":
        description += "The bladder cancer substage is missing. "

    # grade
    if row['grade'] == "G2":
        description += "The tumor grade is G2, which means it is moderately differentiated. "
    elif row['grade'] == "G3":
        description += "The tumor grade is G3, which means it is poorly differentiated. "
    elif row['grade'] == "Missing":
        description += "The tumor grade is missing. "

    # reTUR
    if row['reTUR'] == "Yes":
        description += "A re-transurethral resection (reTUR) was performed before initiating Bacillus Calmette-Guérin (BCG) therapy. "
    elif row['reTUR'] == "No":
        description += "No re-transurethral resection (reTUR) was performed before initiating Bacillus Calmette-Guérin (BCG) therapy. "
    elif row['reTUR'] == "Missing":
        description += "Information on whether a re-transurethral resection (reTUR) was performed prior to Bacillus Calmette-Guérin (BCG) therapy is missing. "

    # LVI
    if row['LVI'] == "Yes":
        description += "Lymphovascular invasion was observed on the Hematoxylin and Eosin (H&E) stained slide. "
    elif row['LVI'] == "No":
        description += "No lymphovascular invasion was observed on the Hematoxylin and Eosin (H&E) stained slide. "
    elif row['LVI'] == "Missing":
        description += "Information about lymphovascular invasion on the Hematoxylin and Eosin (H&E) stained slide is missing. "

    # variant
    if row['variant'] == "UCC":
        description += "Tumor histology reveals urothelial carcinoma with typical morphological features. "
    elif row['variant'] == "UCC + Variant":
        description += "Tumor histology reveals urothelial carcinoma with additional variant histological patterns. "
    elif row['variant'] == "Missing":
        description += "Information about tumor histology variant is missing. "

    # EORTC
    if row['EORTC'] == "High risk":
        description += "According to the European Organization for Research and Treatment of Cancer (EORTC) classification, the patient is at high risk. "
    elif row['EORTC'] == "Highest risk":
        description += "According to the European Organization for Research and Treatment of Cancer (EORTC) classification, the patient is at the highest risk level. "
    elif row['EORTC'] == "Missing":
        description += "The patient's risk classification according to the European Organization for Research and Treatment of Cancer (EORTC) is missing. "

    # no_instillations
    if row['no_instillations'] != "Missing":
        description += f"The patient received a total of {int(row['no_instillations'])} Bacillus Calmette-Guérin (BCG) instillations, an immunotherapy used for bladder cancer. "
    else:
        description += "The number of Bacillus Calmette-Guérin (BCG) instillations, an immunotherapy for bladder cancer, is missing. "

    return description


def get_clinical_embedding(clinical_text: str, model_path: Path) -> np.ndarray:
    # Task-specific instruction
    task_instruction = "Given a patient's clinical history, tumor characteristics, and treatment information, extract key information to predict their Bacillus Calmette-Guérin (BCG) Response Subtype (BRS1, BRS2, or BRS3). These biomarker-derived subtypes indicate how the patient is likely to respond to BCG immunotherapy."
    # Format according to E5 instruct style
    formatted_text = f"Instruct: {task_instruction}\nQuery: {clinical_text}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(str(model_path), device=device)

    embedding = model.encode(
        [formatted_text],
        convert_to_tensor=False,
        normalize_embeddings=True
    )
    del model
    torch.cuda.empty_cache()
    
    return np.array(embedding[0])
