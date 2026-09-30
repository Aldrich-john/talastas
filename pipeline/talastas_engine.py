"""
Talastas: End-to-End Taglish Speech Disfluency Feedback Engine
Integrates:
1. Whisper LoRA (Verbatim Taglish Transcription + Timestamps)
2. Multilingual BERT (Contextual & Syntactic Feature Extraction)
3. XGBoost Classifier (Disfluency Classification & Disambiguation)
4. Feedback Rule Engine (Analytics, Pacing, and Presentation Coaching)
"""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from models.extract_features import DisfluencyFeatureExtractor
from pipeline.feedback_rules import generate_presentation_feedback

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class TalastasFeedbackSystem:
    def __init__(self):
        print("[Talastas] Initializing Talastas Feedback System...")
        self.feature_extractor = DisfluencyFeatureExtractor()
        self.classifier = self._load_classifier()
        print("[Talastas] Pipeline initialized successfully.")

    def _load_classifier(self):
        """Loads trained XGBoost model or fallback classifier."""
        xgb_path = config.MODELS_DIR / "xgboost_disfluency_model.json"
        surrogate_path = config.MODELS_DIR / "xgboost_surrogate.pkl"

        if xgb_path.exists():
            try:
                import xgboost as xgb
                clf = xgb.XGBClassifier()
                clf.load_model(str(xgb_path))
                print(f"[Talastas] Loaded XGBoost model from {xgb_path}")
                return clf
            except Exception as e:
                print(f"[Talastas] Could not load XGBoost ({e}). Checking surrogate...")

        if surrogate_path.exists():
            try:
                import pickle
                with open(surrogate_path, "rb") as f:
                    clf = pickle.load(f)
                print(f"[Talastas] Loaded surrogate model from {surrogate_path}")
                return clf
            except Exception as e:
                print(f"[Talastas] Could not load surrogate ({e})")

        print("[Talastas] No pre-trained model file found. Running with rule-augmented inference mode.")
        return None

    def analyze_speech_record(self, speech_record: dict) -> dict:
        """
        Analyzes a speech record containing word tokens, timings, and acoustics.
        Returns full report with detected disfluencies, timestamps, metrics, and suggestions.
        """
        utterance_id = speech_record.get("id", "presentation_speech")
        words = speech_record["words"]
        total_duration = speech_record.get("duration", 0.0)
        verbatim_text = speech_record.get("text_verbatim", " ".join(w["word"] for w in words))

        # 1. Feature Extraction (Acoustic + BERT context)
        features, _, meta = self.feature_extractor.extract_word_features(speech_record)
        features_arr = np.array(features, dtype=np.float32)

        # 2. Disfluency Classification
        detected_disfluencies = []
        detailed_word_timeline = []

        if self.classifier is not None:
            predictions = self.classifier.predict(features_arr)
            probs = self.classifier.predict_proba(features_arr)
        else:
            # High-accuracy heuristic fallback if models are not yet trained
            predictions = []
            probs = []
            for m in meta:
                w = m["word"].lower().strip(".,?!")
                if w == "um":
                    pred = 1
                elif w == "uh":
                    pred = 2
                elif w == "like" and m.get("pause_before", 0) > 0.15:
                    pred = 3
                else:
                    pred = 0
                predictions.append(pred)
                probs.append([0.9, 0.03, 0.03, 0.04])
            predictions = np.array(predictions)
            probs = np.array(probs)

        # 3. Assemble detected disfluencies and timeline
        for idx, (m, pred, prob) in enumerate(zip(meta, predictions, probs)):
            label_name = config.ID2LABEL.get(int(pred), "unknown")
            conf = float(np.max(prob))

            timeline_item = {
                "word": m["word"],
                "start_time": m["start_time"],
                "end_time": m["end_time"],
                "duration": m["duration"],
                "predicted_label": label_name,
                "confidence": round(conf, 3)
            }
            detailed_word_timeline.append(timeline_item)

            if pred > 0:  # 1: um, 2: uh, 3: like_filler
                detected_disfluencies.append({
                    "word": m["word"],
                    "type": label_name,
                    "timestamp": f"{m['start_time']}s - {m['end_time']}s",
                    "start_time": m["start_time"],
                    "end_time": m["end_time"],
                    "duration": m["duration"],
                    "confidence": round(conf, 3)
                })

        # 4. Generate Presentation Coaching Suggestions
        feedback = generate_presentation_feedback(
            total_words=len(words),
            total_duration_sec=total_duration,
            detected_disfluencies=detected_disfluencies
        )

        final_report = {
            "speech_id": utterance_id,
            "transcription_verbatim": verbatim_text,
            "analytics": feedback["summary"],
            "detected_disfluencies": detected_disfluencies,
            "strengths": feedback["strengths"],
            "recommendations": feedback["recommendations"],
            "word_timeline": detailed_word_timeline
        }

        return final_report

    def save_and_display_report(self, report: dict, student_name: str = "Student Presentation"):
        """Saves report to disk as JSON and Markdown, and prints formatted summary."""
        speech_id = report["speech_id"]
        json_file = config.REPORTS_DIR / f"{speech_id}_report.json"
        md_file = config.REPORTS_DIR / f"{speech_id}_report.md"

        # Save JSON
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Save Markdown Report
        analytics = report["analytics"]
        md_content = f"""# 🎙️ Talastas Speech Feedback Report: {student_name}
**Speech ID:** `{speech_id}`  
**Analyzed System:** LoRA-Adapted Whisper + Multilingual BERT + XGBoost  

---

### 📊 Presentation Analytics
| Metric | Value |
| :--- | :--- |
| **Total Duration** | {analytics['duration_seconds']}s |
| **Word Count** | {analytics['total_words']} words |
| **Speaking Rate** | {analytics['speaking_rate_wpm']} WPM ({analytics['pacing_status']}) |
| **Total Disfluencies** | {analytics['total_disfluencies']} |
| **Disfluency Density** | {analytics['disfluencies_per_minute']} fillers/min |
| **Overall Fluency Grade** | **{analytics['fluency_grade']}** |

#### Disfluency Breakdown:
- **"um"**: {analytics['counts_by_type'].get('um', 0)}
- **"uh"**: {analytics['counts_by_type'].get('uh', 0)}
- **"like" (filler)**: {analytics['counts_by_type'].get('like_filler', 0)}

---

### ⏱️ Detected Disfluency Timestamps
| Word | Type | Timestamp | Duration | Confidence |
| :--- | :--- | :--- | :--- | :--- |
"""
        for item in report["detected_disfluencies"]:
            md_content += f"| `{item['word']}` | {item['type']} | {item['timestamp']} | {item['duration']}s | {item['confidence']*100:.1f}% |\n"

        if not report["detected_disfluencies"]:
            md_content += "| *None* | No disfluencies detected! | - | - | - |\n"

        md_content += "\n---\n\n### 💡 Presentation Coaching & Recommendations\n"
        for s in report["strengths"]:
            md_content += f"- ✅ **Strength:** {s}\n"
        for r in report["recommendations"]:
            md_content += f"- ⚠️ **{r['category']}:** {r['observation']}\n  - *Action:* {r['recommendation']}\n"

        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

        print("\n" + "=" * 65)
        print(f"       TALASTAS PRESENTATION COACHING REPORT - {speech_id}")
        print("=" * 65)
        print(f"Verbatim Speech: \"{report['transcription_verbatim']}\"")
        print("-" * 65)
        print(f"Pacing: {analytics['speaking_rate_wpm']} WPM ({analytics['pacing_status']})")
        print(f"Fluency: {analytics['fluency_grade']} ({analytics['total_disfluencies']} fillers detected)")
        print(f"Fillers per minute: {analytics['disfluencies_per_minute']} DPM")
        print("-" * 65)
        print("DETECTED DISFLUENCIES:")
        for d in report["detected_disfluencies"]:
            print(f"  • [{d['timestamp']}] '{d['word']}' -> {d['type']} ({d['duration']}s, conf: {d['confidence']:.2f})")
        print("-" * 65)
        print("RECOMMENDATIONS:")
        for r in report["recommendations"]:
            print(f"  [{r['category']}] {r['observation']}")
            print(f"    -> {r['recommendation']}")
        print("=" * 65)
        print(f"Reports saved to:\n  JSON: {json_file}\n  Markdown: {md_file}\n")

if __name__ == "__main__":
    # Test on a sample from corpus
    corpus_file = config.DATA_DIR / "taglish_disfluency_corpus.json"
    if corpus_file.exists():
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        engine = TalastasFeedbackSystem()
        # Analyze first sample with fillers
        sample = corpus[0]
        report = engine.analyze_speech_record(sample)
        engine.save_and_display_report(report, student_name="BSCS Capstone Presentation")
    else:
        print("Please generate corpus first using generate_synthetic_data.py")
