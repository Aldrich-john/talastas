"""
Artic: Complete Demonstration Runner
Executes the full pipeline step-by-step:
1. Synthetic Taglish data generation
2. LoRA-Adapted Whisper training check
3. Acoustic & BERT feature extraction
4. XGBoost training & thesis evaluation metrics (Accuracy, Precision, Recall, F1)
5. Disambiguation test:
   - "I really like the methodology..." (Lexical verb -> NOT a filler)
   - "...kasi like it is really important" (Disfluent filler -> FLAGGED)
6. Presentation report & coaching suggestion generation
"""

import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.generate_synthetic_data import create_dataset
from models.train_whisper_lora import run_fine_tuning_dummy
from models.extract_features import build_dataset_features
from models.train_xgboost import train_xgboost_classifier
from pipeline.artic_engine import ArticFeedbackSystem
import config

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_artic_demo():
    print("\n" + "=" * 70)
    print("      [STARTING ARTIC DISFLUENCY FEEDBACK SYSTEM DEMO]")
    print("      (LoRA-Whisper + Multilingual BERT + XGBoost)")
    print("=" * 70)

    # 1. Dataset Generation
    corpus_file = config.DATA_DIR / "taglish_disfluency_corpus.json"
    if not corpus_file.exists():
        print("\n[Step 1/5] Generating Taglish corpus with audio waveforms...")
        create_dataset()
    else:
        print("\n[Step 1/5] Taglish corpus found.")

    # 2. Whisper-LoRA Fine-tuning Setup
    print("\n[Step 2/5] Initializing LoRA-Adapted Whisper fine-tuning scaffold...")
    run_fine_tuning_dummy()

    # 3. Acoustic + BERT Feature Extraction
    print("\n[Step 3/5] Extracting Acoustic + BERT contextual features...")
    build_dataset_features()

    # 4. XGBoost Model Training & Metrics
    print("\n[Step 4/5] Training XGBoost Disfluency Classifier & Evaluating Metrics...")
    train_xgboost_classifier()

    # 5. End-to-End Inference & Feedback Generation
    print("\n[Step 5/5] Running End-to-End Inference and Presentation Feedback...")
    engine = ArticFeedbackSystem()

    with open(corpus_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Select two contrasting samples to demonstrate "like" disambiguation:
    # Sample A: contains disfluent "like", "um", "uh"
    sample_a = corpus[0]
    # Sample B: contains lexical "like" ("I really like the methodology...")
    sample_b = corpus[2]

    print("\n" + "#" * 70)
    print("TEST CASE 1: Presentation with multiple disfluencies ('uh', filler 'like')")
    print("#" * 70)
    report_a = engine.analyze_speech_record(sample_a)
    engine.save_and_display_report(report_a, student_name="Student Presentation (With Crutches)")

    print("\n" + "#" * 70)
    print("TEST CASE 2: Presentation with lexical 'like' ('I really like the methodology')")
    print("EXPECTED BEHAVIOR: BERT + XGBoost should NOT flag this 'like' as a filler!")
    print("#" * 70)
    report_b = engine.analyze_speech_record(sample_b)
    engine.save_and_display_report(report_b, student_name="Student Presentation (Fluent Lexical Use)")

    print("\n" + "=" * 70)
    print("      [SUCCESS] ARTIC DEMO COMPLETED SUCCESSFULLY!")
    print("Generated Reports:")
    print(f"  • {config.REPORTS_DIR / (sample_a['id'] + '_report.md')}")
    print(f"  • {config.REPORTS_DIR / (sample_b['id'] + '_report.md')}")
    print("=" * 70)

if __name__ == "__main__":
    run_artic_demo()
