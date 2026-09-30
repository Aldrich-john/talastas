import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure directories exist
for p in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Model Checkpoints & Defaults (All 100% Free & Open-Source)
WHISPER_BASE_MODEL = "openai/whisper-small"     # Free HuggingFace model (or whisper-tiny for fast CPU testing)
WHISPER_FALLBACK_MODEL = "openai/whisper-tiny"
BERT_BASE_MODEL = "bert-base-multilingual-cased" # Multilingual BERT supporting Tagalog + English
TAGALOG_ROBERTA_MODEL = "jcblaise/roberta-tagalog-base" # Optional Philippine NLP community model

# Target disfluency classes according to Artic thesis scope:
# 0: Fluent (normal word)
# 1: "um"
# 2: "uh"
# 3: "like" (used as disfluent filler)
TARGET_DISFLUENCIES = ["fluent", "um", "uh", "like_filler"]
LABEL2ID = {name: i for i, name in enumerate(TARGET_DISFLUENCIES)}
ID2LABEL = {i: name for i, name in enumerate(TARGET_DISFLUENCIES)}

# LoRA Configuration for Whisper
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# XGBoost hyperparameters
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": len(TARGET_DISFLUENCIES),
    "max_depth": 5,
    "learning_rate": 0.08,
    "n_estimators": 120,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42
}
