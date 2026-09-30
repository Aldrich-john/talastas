# 🎙️ Talastas: Taglish Speech Disfluency Feedback System
**Utilizing LoRA-Adapted Whisper, BERT-Based Features, and XGBoost for Presentation Training**

---

## 📌 Executive Summary & Model Cost Analysis

> **Are the proposed models free?**  
> **YES. All three core models and libraries are 100% FREE and Open-Source.**  
> You do **not** need paid APIs (like OpenAI ChatGPT API or Google Cloud Speech API). The entire pipeline runs locally or on free Google Colab GPUs (T4).

| Component | Technology | License | Cost | Role in Talastas |
| :--- | :--- | :--- | :--- | :--- |
| **ASR & Alignment** | `openai/whisper-small` + `peft` (LoRA) | MIT / Apache 2.0 | **100% Free** | Verbatim Taglish transcription with disfluency preservation ("um", "uh", "like") & word timestamps. |
| **Contextual NLP** | `bert-base-multilingual-cased` or `jcblaise/roberta-tagalog-base` | Apache 2.0 | **100% Free** | Syntactic & contextual embeddings to disambiguate filler "like" vs lexical verb/preposition "like". |
| **Disfluency Classifier** | `xgboost` (eXtreme Gradient Boosting) | Apache 2.0 | **100% Free** | Multi-class tabular classification fusing acoustic prosody + BERT embeddings. |
| **Presentation Feedback** | Rule-Based Coaching Engine | Open Source | **100% Free** | Converts disfluency metrics (DPM, WPM) into presentation-oriented coaching advice. |

---

## 🏗️ System Architecture

```
                  ┌─────────────────────────────────────────┐
                  │ Recorded Taglish Audio (.wav / .mp3)    │
                  └──────────────────┬──────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────────┐
                  │ 1. LoRA-Adapted Whisper ASR             │
                  │    - Retains verbatim "um", "uh", "like"│
                  │    - Extracts word-level timestamps     │
                  └──────────────────┬──────────────────────┘
                                     │
               ┌─────────────────────┴──────────────────────┐
               ▼                                            ▼
┌──────────────────────────────┐             ┌──────────────────────────────┐
│ 2a. Acoustic Features        │             │ 2b. Multilingual BERT        │
│     - Preceding silence      │             │     - Contextual embedding   │
│     - Following pause        │             │     - Syntactic disambiguator│
│     - Elongated duration     │             │       ("I like" vs ",like,") │
└──────────────┬───────────────┘             └──────────────┬───────────────┘
               └─────────────────────┬──────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────────┐
                  │ 3. XGBoost Disfluency Classifier        │
                  │    Outputs: [Fluent, um, uh, like]      │
                  │    Evaluates: Precision, Recall, F1     │
                  └──────────────────┬──────────────────────┘
                                     │
                                     ▼
                  ┌─────────────────────────────────────────┐
                  │ 4. Analytics & Feedback Engine          │
                  │    - Timestamps & disfluency counts     │
                  │    - Disfluency density (DPM) & WPM     │
                  │    - Rule-based presentation suggestions│
                  │    - ISO/IEC 25010 compliant report     │
                  └─────────────────────────────────────────┘
```

---

## 🚀 Quickstart: Running the Complete Prototype

### 1. Installation
Clone or navigate to the directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the End-to-End Demo
Run the single command below to execute the entire 5-stage pipeline:
```bash
python demo_run.py
```

This will automatically:
1. Generate an annotated Taglish speech corpus with audio waveforms in `data/`.
2. Configure the LoRA adapters (`r=16, alpha=32`) for Whisper in `saved_models/whisper_lora_taglish`.
3. Extract acoustic pause/duration features combined with BERT contextual representations.
4. Train the XGBoost disfluency classifier, printing **Precision, Recall, F1-Score, and Confusion Matrix**.
5. Run the **Disambiguation Test**:
   - Utterance with fillers: `"Good morning... uh we are going to present kasi like it is really important"` ➡️ Flags **uh** and **like (filler)**.
   - Utterance with lexical verb: `"I really like the methodology because it solves the bottleneck..."` ➡️ **Correctly recognizes "like" as a verb and does NOT flag it!**
6. Generate structured student coaching reports in `reports/` as JSON and Markdown.

---

## 📂 Project Structure

```
finetune/
├── config.py                       # Central paths, hyperparams, and model IDs
├── requirements.txt                # Free open-source Python dependencies
├── README.md                       # Complete documentation & study mapping
├── demo_run.py                     # Master execution runner
│
├── data/
│   ├── generate_synthetic_data.py  # Generates annotated Taglish audio & transcripts
│   ├── taglish_disfluency_corpus.json
│   ├── features_dataset.npz        # Tabular acoustic + BERT features
│   └── audio_samples/              # Synthetic WAV files
│
├── models/
│   ├── train_whisper_lora.py       # Fine-tunes Whisper with PEFT LoRA
│   ├── extract_features.py         # Acoustic + BERT contextual feature extractor
│   └── train_xgboost.py            # Trains XGBoost and outputs F1/Precision/Recall
│
├── pipeline/
│   ├── talastas_engine.py             # Inference pipeline & report generator
│   └── feedback_rules.py           # Presentation coaching rules & scoring
│
└── reports/                        # Generated student feedback reports
    ├── sample_001_rep0_report.md
    └── model_evaluation_metrics.json
```

---

## 🔬 How Each Thesis Objective is Addressed

### Objective 1: LoRA-Whisper + BERT + XGBoost Disfluency Pipeline
- Standard Whisper automatically cleans and deletes "um" and "uh". Fine-tuning with **LoRA** (`models/train_whisper_lora.py`) conditions the decoder to produce verbatim transcripts while preserving code-switched Taglish phonetic tokens.
- **Multilingual BERT** (`models/extract_features.py`) resolves lexical ambiguity for words like `"like"`:
  - *"I **like** this topic"* (Verb: preceded by pronoun `I`) ➡️ `fluent`
  - *"It was, **like**, sobrang hirap"* (Filler: hesitation pause + clause starter) ➡️ `like_filler`
- **XGBoost** (`models/train_xgboost.py`) fuses acoustic signals (`pause_before`, `pause_after`, `duration`) with BERT representations.

### Objective 2: Data Analytics & Report Generation
- Extracts exact start and end timestamps (e.g. `1.32s - 1.85s`) for each detected crutch.
- Calculates Disfluencies Per Minute (**DPM**) and Speaking Rate (**WPM**).
- Saves both machine-readable JSON records and student-friendly Markdown presentation reports (`reports/`).

### Objective 3 & 4: Evaluation & ISO/IEC 25010 Compliance
- Evaluates classification metrics: **Precision, Recall, F1-Score, and Accuracy** per filler category.
- Adheres to ISO/IEC 25010:2023:
  - *Functional Suitability*: Accurate detection of specified disfluencies ("um", "uh", "like").
  - *Performance Efficiency*: Fast CPU/GPU inference via LoRA + XGBoost.
  - *Usability*: Actionable rule-based presentation coaching tips for students and professionals.
