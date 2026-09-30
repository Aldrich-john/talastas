"""
Synthetic Taglish Dataset Generator for Artic
Generates annotated Taglish utterances containing both fluent words and disfluencies:
- "um"
- "uh"
- "like" (as filler) vs "like" (as lexical verb/preposition, e.g., "I like this idea")
Also produces lightweight synthetic WAV audio files to test the audio processing pipeline.
"""

import json
import random
import wave
import struct
import math
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR

RAW_AUDIO_DIR = DATA_DIR / "audio_samples"
RAW_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Realistic Taglish presentation speech templates
# Each item has (word, is_filler_label: 0=fluent, 1=um, 2=uh, 3=like_filler)
SAMPLE_UTTERANCES = [
    # ===== SPEAKER A: Capstone presentations with mixed fillers =====
    {
        "id": "sample_001",
        "speaker": "speaker_A",
        "text_raw": "Good morning everyone so today uh we are going to present our capstone project kasi like it is really important.",
        "tokens": [
            ("Good", 0), ("morning", 0), ("everyone", 0), ("so", 0), ("today", 0),
            ("uh", 2),
            ("we", 0), ("are", 0), ("going", 0), ("to", 0), ("present", 0), ("our", 0),
            ("capstone", 0), ("project", 0), ("kasi", 0),
            ("like", 3), # filler
            ("it", 0), ("is", 0), ("really", 0), ("important", 0)
        ]
    },
    {
        "id": "sample_002",
        "speaker": "speaker_A",
        "text_raw": "Ang main problem namin dito is um hindi accessible ang data para sa mga students.",
        "tokens": [
            ("Ang", 0), ("main", 0), ("problem", 0), ("namin", 0), ("dito", 0), ("is", 0),
            ("um", 1), # filler
            ("hindi", 0), ("accessible", 0), ("ang", 0), ("data", 0), ("para", 0), ("sa", 0), ("mga", 0), ("students", 0)
        ]
    },
    {
        "id": "sample_003",
        "speaker": "speaker_B",
        "text_raw": "I really like the methodology because it solves the bottleneck sa aming processing.",
        "tokens": [
            ("I", 0), ("really", 0),
            ("like", 0), # lexical verb (NOT a filler!)
            ("the", 0), ("methodology", 0), ("because", 0), ("it", 0), ("solves", 0), ("the", 0),
            ("bottleneck", 0), ("sa", 0), ("aming", 0), ("processing", 0)
        ]
    },
    {
        "id": "sample_004",
        "speaker": "speaker_B",
        "text_raw": "Pagdating sa results, uh we observed na like 85 percent of the users preferred the new UI.",
        "tokens": [
            ("Pagdating", 0), ("sa", 0), ("results", 0),
            ("uh", 2), # filler
            ("we", 0), ("observed", 0), ("na", 0),
            ("like", 3), # filler
            ("85", 0), ("percent", 0), ("of", 0), ("the", 0), ("users", 0), ("preferred", 0), ("the", 0), ("new", 0), ("UI", 0)
        ]
    },
    {
        "id": "sample_005",
        "speaker": "speaker_C",
        "text_raw": "So um basically, ang conclusion po namin is like we need to improve the inference speed.",
        "tokens": [
            ("So", 0),
            ("um", 1), # filler
            ("basically", 0), ("ang", 0), ("conclusion", 0), ("po", 0), ("namin", 0), ("is", 0),
            ("like", 3), # filler
            ("we", 0), ("need", 0), ("to", 0), ("improve", 0), ("the", 0), ("inference", 0), ("speed", 0)
        ]
    },
    {
        "id": "sample_006",
        "speaker": "speaker_C",
        "text_raw": "Kailangan natin tingnan ang user feedback, uh specifically on how like they navigate the menus.",
        "tokens": [
            ("Kailangan", 0), ("natin", 0), ("tingnan", 0), ("ang", 0), ("user", 0), ("feedback", 0),
            ("uh", 2), # filler
            ("specifically", 0), ("on", 0), ("how", 0),
            ("like", 3), # filler
            ("they", 0), ("navigate", 0), ("the", 0), ("menus", 0)
        ]
    },
    {
        "id": "sample_007",
        "speaker": "speaker_D",
        "text_raw": "They look like standard evaluation metrics for our Taglish test corpus.",
        "tokens": [
            ("They", 0), ("look", 0),
            ("like", 0), # preposition (NOT a filler!)
            ("standard", 0), ("evaluation", 0), ("metrics", 0), ("for", 0), ("our", 0), ("Taglish", 0), ("test", 0), ("corpus", 0)
        ]
    },
    {
        "id": "sample_008",
        "speaker": "speaker_D",
        "text_raw": "Next slide please, um we will now discuss the ISO 25010 compliance testing.",
        "tokens": [
            ("Next", 0), ("slide", 0), ("please", 0),
            ("um", 1), # filler
            ("we", 0), ("will", 0), ("now", 0), ("discuss", 0), ("the", 0), ("ISO", 0), ("25010", 0), ("compliance", 0), ("testing", 0)
        ]
    },

    # ===== SPEAKER E: Technical defense presentations =====
    {
        "id": "sample_009",
        "speaker": "speaker_E",
        "text_raw": "So para sa aming system architecture um gumamit kami ng microservices kasi like it scales better.",
        "tokens": [
            ("So", 0), ("para", 0), ("sa", 0), ("aming", 0), ("system", 0), ("architecture", 0),
            ("um", 1),
            ("gumamit", 0), ("kami", 0), ("ng", 0), ("microservices", 0), ("kasi", 0),
            ("like", 3),  # filler
            ("it", 0), ("scales", 0), ("better", 0)
        ]
    },
    {
        "id": "sample_010",
        "speaker": "speaker_E",
        "text_raw": "Uh actually the response time improved by 40 percent after we deployed the cache layer.",
        "tokens": [
            ("Uh", 2),
            ("actually", 0), ("the", 0), ("response", 0), ("time", 0), ("improved", 0), ("by", 0),
            ("40", 0), ("percent", 0), ("after", 0), ("we", 0), ("deployed", 0), ("the", 0), ("cache", 0), ("layer", 0)
        ]
    },
    {
        "id": "sample_011",
        "speaker": "speaker_E",
        "text_raw": "We would like to thank our adviser for guiding us throughout this research.",
        "tokens": [
            ("We", 0), ("would", 0),
            ("like", 0),  # lexical verb (NOT a filler!)
            ("to", 0), ("thank", 0), ("our", 0), ("adviser", 0), ("for", 0), ("guiding", 0),
            ("us", 0), ("throughout", 0), ("this", 0), ("research", 0)
        ]
    },
    {
        "id": "sample_012",
        "speaker": "speaker_F",
        "text_raw": "Um ang dataset namin consists of um 500 annotated Taglish speech recordings from uh volunteer students.",
        "tokens": [
            ("Um", 1),
            ("ang", 0), ("dataset", 0), ("namin", 0), ("consists", 0), ("of", 0),
            ("um", 1),
            ("500", 0), ("annotated", 0), ("Taglish", 0), ("speech", 0), ("recordings", 0), ("from", 0),
            ("uh", 2),
            ("volunteer", 0), ("students", 0)
        ]
    },
    {
        "id": "sample_013",
        "speaker": "speaker_F",
        "text_raw": "The panelists said they like our approach to handling code-switching in the corpus.",
        "tokens": [
            ("The", 0), ("panelists", 0), ("said", 0), ("they", 0),
            ("like", 0),  # lexical verb (NOT a filler!)
            ("our", 0), ("approach", 0), ("to", 0), ("handling", 0), ("code-switching", 0),
            ("in", 0), ("the", 0), ("corpus", 0)
        ]
    },

    # ===== SPEAKER G: Thesis defense with heavy fillers =====
    {
        "id": "sample_014",
        "speaker": "speaker_G",
        "text_raw": "So um uh basically ang aming research is about um detecting disfluencies like sa Taglish speech.",
        "tokens": [
            ("So", 0),
            ("um", 1),
            ("uh", 2),
            ("basically", 0), ("ang", 0), ("aming", 0), ("research", 0), ("is", 0), ("about", 0),
            ("um", 1),
            ("detecting", 0), ("disfluencies", 0),
            ("like", 3),  # filler
            ("sa", 0), ("Taglish", 0), ("speech", 0)
        ]
    },
    {
        "id": "sample_015",
        "speaker": "speaker_G",
        "text_raw": "Yung accuracy namin is like 95 percent which is uh higher than the baseline model.",
        "tokens": [
            ("Yung", 0), ("accuracy", 0), ("namin", 0), ("is", 0),
            ("like", 3),  # filler
            ("95", 0), ("percent", 0), ("which", 0), ("is", 0),
            ("uh", 2),
            ("higher", 0), ("than", 0), ("the", 0), ("baseline", 0), ("model", 0)
        ]
    },
    {
        "id": "sample_016",
        "speaker": "speaker_G",
        "text_raw": "It looks like the training converged after 50 epochs sa aming GPU server.",
        "tokens": [
            ("It", 0), ("looks", 0),
            ("like", 0),  # preposition (NOT a filler!)
            ("the", 0), ("training", 0), ("converged", 0), ("after", 0), ("50", 0), ("epochs", 0),
            ("sa", 0), ("aming", 0), ("GPU", 0), ("server", 0)
        ]
    },

    # ===== SPEAKER H: Research methodology presentations =====
    {
        "id": "sample_017",
        "speaker": "speaker_H",
        "text_raw": "For the methodology namin um ginamit namin ang mixed methods approach para sa data collection.",
        "tokens": [
            ("For", 0), ("the", 0), ("methodology", 0), ("namin", 0),
            ("um", 1),
            ("ginamit", 0), ("namin", 0), ("ang", 0), ("mixed", 0), ("methods", 0), ("approach", 0),
            ("para", 0), ("sa", 0), ("data", 0), ("collection", 0)
        ]
    },
    {
        "id": "sample_018",
        "speaker": "speaker_H",
        "text_raw": "Uh ang respondents namin are like 200 college students na nag-present ng thesis defense.",
        "tokens": [
            ("Uh", 2),
            ("ang", 0), ("respondents", 0), ("namin", 0), ("are", 0),
            ("like", 3),  # filler (approximation/hedging)
            ("200", 0), ("college", 0), ("students", 0), ("na", 0), ("nag-present", 0),
            ("ng", 0), ("thesis", 0), ("defense", 0)
        ]
    },
    {
        "id": "sample_019",
        "speaker": "speaker_H",
        "text_raw": "I like how the framework handles both Filipino and English tokens seamlessly.",
        "tokens": [
            ("I", 0),
            ("like", 0),  # lexical verb (NOT a filler!)
            ("how", 0), ("the", 0), ("framework", 0), ("handles", 0), ("both", 0),
            ("Filipino", 0), ("and", 0), ("English", 0), ("tokens", 0), ("seamlessly", 0)
        ]
    },

    # ===== SPEAKER I: Q&A defense responses =====
    {
        "id": "sample_020",
        "speaker": "speaker_I",
        "text_raw": "That is a good question po um so basically kasi like we considered na rin yan sa design phase.",
        "tokens": [
            ("That", 0), ("is", 0), ("a", 0), ("good", 0), ("question", 0), ("po", 0),
            ("um", 1),
            ("so", 0), ("basically", 0), ("kasi", 0),
            ("like", 3),  # filler
            ("we", 0), ("considered", 0), ("na", 0), ("rin", 0), ("yan", 0), ("sa", 0), ("design", 0), ("phase", 0)
        ]
    },
    {
        "id": "sample_021",
        "speaker": "speaker_I",
        "text_raw": "Uh to answer your question ang limitation po namin is uh yung size ng training data.",
        "tokens": [
            ("Uh", 2),
            ("to", 0), ("answer", 0), ("your", 0), ("question", 0), ("ang", 0),
            ("limitation", 0), ("po", 0), ("namin", 0), ("is", 0),
            ("uh", 2),
            ("yung", 0), ("size", 0), ("ng", 0), ("training", 0), ("data", 0)
        ]
    },
    {
        "id": "sample_022",
        "speaker": "speaker_I",
        "text_raw": "The results look like what we expected based on the literature review.",
        "tokens": [
            ("The", 0), ("results", 0), ("look", 0),
            ("like", 0),  # preposition (NOT a filler!)
            ("what", 0), ("we", 0), ("expected", 0), ("based", 0), ("on", 0),
            ("the", 0), ("literature", 0), ("review", 0)
        ]
    },

    # ===== SPEAKER J: Demo and walkthrough presentations =====
    {
        "id": "sample_023",
        "speaker": "speaker_J",
        "text_raw": "So dito sa demo namin um you can see na like the system detects um yung mga fillers in real time.",
        "tokens": [
            ("So", 0), ("dito", 0), ("sa", 0), ("demo", 0), ("namin", 0),
            ("um", 1),
            ("you", 0), ("can", 0), ("see", 0), ("na", 0),
            ("like", 3),  # filler
            ("the", 0), ("system", 0), ("detects", 0),
            ("um", 1),
            ("yung", 0), ("mga", 0), ("fillers", 0), ("in", 0), ("real", 0), ("time", 0)
        ]
    },
    {
        "id": "sample_024",
        "speaker": "speaker_J",
        "text_raw": "Uh kapag nag-click ka dito makikita mo yung uh detailed breakdown ng each utterance.",
        "tokens": [
            ("Uh", 2),
            ("kapag", 0), ("nag-click", 0), ("ka", 0), ("dito", 0), ("makikita", 0), ("mo", 0), ("yung", 0),
            ("uh", 2),
            ("detailed", 0), ("breakdown", 0), ("ng", 0), ("each", 0), ("utterance", 0)
        ]
    },

    # ===== SPEAKER K: Academic paper presentations =====
    {
        "id": "sample_025",
        "speaker": "speaker_K",
        "text_raw": "In our related literature um maraming studies na ang nag-focus sa English disfluency detection.",
        "tokens": [
            ("In", 0), ("our", 0), ("related", 0), ("literature", 0),
            ("um", 1),
            ("maraming", 0), ("studies", 0), ("na", 0), ("ang", 0), ("nag-focus", 0),
            ("sa", 0), ("English", 0), ("disfluency", 0), ("detection", 0)
        ]
    },
    {
        "id": "sample_026",
        "speaker": "speaker_K",
        "text_raw": "However wala pa masyadong research na like specifically targets Taglish code-switching scenarios.",
        "tokens": [
            ("However", 0), ("wala", 0), ("pa", 0), ("masyadong", 0), ("research", 0), ("na", 0),
            ("like", 3),  # filler
            ("specifically", 0), ("targets", 0), ("Taglish", 0), ("code-switching", 0), ("scenarios", 0)
        ]
    },
    {
        "id": "sample_027",
        "speaker": "speaker_K",
        "text_raw": "We like to emphasize na ang contribution namin is the Taglish-specific feature set.",
        "tokens": [
            ("We", 0),
            ("like", 0),  # lexical verb (NOT a filler!)
            ("to", 0), ("emphasize", 0), ("na", 0), ("ang", 0), ("contribution", 0), ("namin", 0),
            ("is", 0), ("the", 0), ("Taglish-specific", 0), ("feature", 0), ("set", 0)
        ]
    },

    # ===== SPEAKER L: Nervous presenter with many fillers =====
    {
        "id": "sample_028",
        "speaker": "speaker_L",
        "text_raw": "Um so um ang aming recommendation po is uh to expand the dataset and um retrain the model.",
        "tokens": [
            ("Um", 1),
            ("so", 0),
            ("um", 1),
            ("ang", 0), ("aming", 0), ("recommendation", 0), ("po", 0), ("is", 0),
            ("uh", 2),
            ("to", 0), ("expand", 0), ("the", 0), ("dataset", 0), ("and", 0),
            ("um", 1),
            ("retrain", 0), ("the", 0), ("model", 0)
        ]
    },
    {
        "id": "sample_029",
        "speaker": "speaker_L",
        "text_raw": "Kasi like uh ang problem is like hindi siya robust enough para sa noisy audio environments.",
        "tokens": [
            ("Kasi", 0),
            ("like", 3),  # filler
            ("uh", 2),
            ("ang", 0), ("problem", 0), ("is", 0),
            ("like", 3),  # filler (second instance)
            ("hindi", 0), ("siya", 0), ("robust", 0), ("enough", 0),
            ("para", 0), ("sa", 0), ("noisy", 0), ("audio", 0), ("environments", 0)
        ]
    },
    {
        "id": "sample_030",
        "speaker": "speaker_L",
        "text_raw": "It feels like a natural extension of existing speech processing research sa Philippines.",
        "tokens": [
            ("It", 0), ("feels", 0),
            ("like", 0),  # preposition (NOT a filler!)
            ("a", 0), ("natural", 0), ("extension", 0), ("of", 0), ("existing", 0),
            ("speech", 0), ("processing", 0), ("research", 0), ("sa", 0), ("Philippines", 0)
        ]
    },
]

def generate_synthetic_audio(file_path: Path, duration_seconds: float = 3.5, sample_rate: int = 16000):
    """
    Generates a harmless modulated synthetic WAV audio file simulating human vocal tone & pauses.
    Allows testing librosa/soundfile/Whisper audio loaders without external downloads.
    """
    n_samples = int(duration_seconds * sample_rate)
    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        for i in range(n_samples):
            t = i / sample_rate
            # Synthesize fundamental frequency (150-220Hz speech range) with harmonic envelope
            freq = 180 + 30 * math.sin(2 * math.pi * 1.5 * t)
            envelope = 0.5 * (1 + math.sin(2 * math.pi * 3.0 * t))
            # Simulate slight pause
            if 1.0 < (t % 3.0) < 1.35:
                envelope = 0.02
            val = int(32767.0 * 0.3 * envelope * math.sin(2 * math.pi * freq * t))
            wav_file.writeframesraw(struct.pack("<h", max(-32767, min(32767, val))))

def create_dataset():
    """Generates the synthetic dataset metadata, word alignments, and WAV files."""
    dataset_records = []
    
    # Expand samples with randomized acoustic properties to create a solid training set
    expanded_samples = []
    for rep in range(5):  # 30 templates * 5 reps = 150 utterances
        for orig in SAMPLE_UTTERANCES:
            item_id = f"{orig['id']}_rep{rep}"
            audio_path = RAW_AUDIO_DIR / f"{item_id}.wav"
            
            # Estimate timestamps and word durations
            current_time = 0.2
            word_entries = []
            
            for word, label in orig["tokens"]:
                # Normal word vs filler acoustics
                if label in [1, 2]:  # um, uh
                    pause_before = round(random.uniform(0.25, 0.70), 3) # long hesitation pause
                    duration = round(random.uniform(0.35, 0.65), 3)     # elongated vowel
                    pause_after = round(random.uniform(0.15, 0.45), 3)
                elif label == 3:      # like as filler
                    pause_before = round(random.uniform(0.15, 0.40), 3)
                    duration = round(random.uniform(0.20, 0.35), 3)
                    pause_after = round(random.uniform(0.10, 0.30), 3)
                else:                 # fluent words (or lexical "like")
                    pause_before = round(random.uniform(0.01, 0.12), 3)
                    duration = round(random.uniform(0.12, 0.32), 3)
                    pause_after = round(random.uniform(0.01, 0.10), 3)

                start_time = round(current_time + pause_before, 3)
                end_time = round(start_time + duration, 3)
                current_time = round(end_time + pause_after, 3)

                word_entries.append({
                    "word": word,
                    "label": label,
                    "label_name": ["fluent", "um", "uh", "like_filler"][label],
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "pause_before": pause_before,
                    "pause_after": pause_after
                })

            total_duration = round(current_time + 0.3, 2)
            generate_synthetic_audio(audio_path, duration_seconds=total_duration)

            expanded_samples.append({
                "id": item_id,
                "audio_path": str(audio_path),
                "duration": total_duration,
                "speaker": orig["speaker"],
                "text_verbatim": orig["text_raw"],
                "words": word_entries
            })

    output_file = DATA_DIR / "taglish_disfluency_corpus.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(expanded_samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(expanded_samples)} sample utterances with audio files.")
    print(f"Corpus saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    create_dataset()
